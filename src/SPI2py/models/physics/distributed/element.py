import jax.numpy as jnp
from jax import vmap
from .quadrature import shape_functions, shape_functions_vec


def assemble_local_stiffness_matrix(element_nodes, k_eff, gauss_pts, gauss_wts):
    """
    Compute the 8x8 stiffness matrix for a hexahedral element in an unvectorized manner.
    """

    # Number of quadrature points.
    n_qp = len(gauss_pts)

    # Initialize the element stiffness matrix.
    Ke = jnp.zeros((8, 8))

    # Loop over quadrature points.
    for i in range(n_qp):
        xi = gauss_pts[i]
        w_xi = gauss_wts[i]

        for j in range(n_qp):
            eta = gauss_pts[j]
            w_eta = gauss_wts[j]

            for k in range(n_qp):
                zeta = gauss_pts[k]
                w_zeta = gauss_wts[k]
                w_total = w_xi * w_eta * w_zeta
                N, dN_dxi = shape_functions(xi, eta, zeta)  # N: (8,), dN_dxi: (8,3)
                J = jnp.zeros((3, 3))
                for i_node in range(8):
                    J = J + jnp.outer(element_nodes[i_node], dN_dxi[i_node])
                detJ = jnp.abs(jnp.linalg.det(J))
                J_inv = jnp.linalg.inv(J)
                dN_dx = jnp.zeros((8, 3))
                for i_node in range(8):
                    dN_dx = dN_dx.at[i_node].set(jnp.matmul(J_inv, dN_dxi[i_node]))
                contrib = k_eff * jnp.matmul(dN_dx, dN_dx.T) * detJ * w_total
                Ke = Ke + contrib
    return Ke


def assemble_local_stiffness_matrix_vec(element_nodes, k_eff, gauss_pts, gauss_wts):
    """
    Compute the 8x8 stiffness matrix for a hexahedral element using manual vectorization.

    Parameters:
      element_nodes: (8,3) array of node coordinates.
      k_eff        : effective thermal conductivity.
      gauss_pts    : 1D array of Gauss quadrature points.
      gauss_wts    : 1D array of Gauss quadrature weights.

    Returns:
      Ke: (8,8) element stiffness matrix.
    """

    # Create a tensor grid of quadrature points.
    xi_grid, eta_grid, zeta_grid = jnp.meshgrid(gauss_pts, gauss_pts, gauss_pts, indexing='ij')
    xi = xi_grid.flatten()
    eta = eta_grid.flatten()
    zeta = zeta_grid.flatten()
    n_qp = xi.shape[0]

    # Build corresponding quadrature weights.
    wx, wy, wz = jnp.meshgrid(gauss_wts, gauss_wts, gauss_wts, indexing='ij')
    w_total = (wx * wy * wz).flatten()  # shape (n_qp,)

    # Evaluate shape functions and their derivatives at all quadrature points.
    N_all, dN_dxi_all = shape_functions_vec(xi, eta, zeta)  # Shapes: (n_qp, 8) and (n_qp, 8, 3)

    # Compute the Jacobian for each quadrature point.
    # We want: J_all[q, j, k] = sum_{i=0}^{7} element_nodes[i, j] * dN_dxi_all[q, i, k]
    J_all = jnp.einsum('ij,qik->qjk', element_nodes, dN_dxi_all)

    # Compute determinants of the Jacobians.
    detJ_all = jnp.abs(jnp.linalg.det(J_all))  # shape (n_qp,)

    # Compute the inverse of each Jacobian.
    J_inv_all = jnp.linalg.inv(J_all)  # shape (n_qp, 3, 3)

    # Map the derivatives from natural to physical space.
    # For each quadrature point and each node: dN_dx = J_inv_all[q] @ dN_dxi_all[q, i]
    # Using jnp.matmul with broadcasting:
    dN_dx_all = jnp.matmul(J_inv_all[:, None, :, :], dN_dxi_all[..., None]).squeeze(-1)
    # dN_dx_all has shape (n_qp, 8, 3)

    # Compute the contribution from each quadrature point.
    contrib_all = jnp.einsum('qik,qjk->qij', dN_dx_all, dN_dx_all)
    contrib_all = k_eff * contrib_all * detJ_all[:, None, None] * w_total[:, None, None]

    # Sum the contributions from all quadrature points.
    Ke = jnp.sum(contrib_all, axis=0)

    return Ke

