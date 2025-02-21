import jax.numpy as jnp
from .quadrature import shape_functions


def assemble_local_stiffness_matrix(element_nodes, k_eff, gauss_pts, gauss_wts):
    """
    Compute the 8x8 element stiffness matrix for a single element.

    Parameters:
      element_nodes: (8, 3) array with the coordinates of the element's nodes.
      k_eff:         Scalar effective conductivity for the element.
      gauss_pts:     1D array of Gauss quadrature points.
      gauss_wts:     1D array of Gauss quadrature weights.

    Returns:
      Ke: (8,8) element stiffness matrix.
    """

    # Build a 3D quadrature grid.
    # Shape: (n_qp,)
    xi_grid, eta_grid, zeta_grid = jnp.meshgrid(gauss_pts, gauss_pts, gauss_pts, indexing='ij')
    xi = xi_grid.flatten()
    eta = eta_grid.flatten()
    zeta = zeta_grid.flatten()

    # Build the total quadrature weights.
    # Shape: (n_qp,)
    wx, wy, wz = jnp.meshgrid(gauss_wts, gauss_wts, gauss_wts, indexing='ij')
    w_total = (wx * wy * wz).flatten()

    # Evaluate shape functions and their derivatives at all quadrature points.
    # Shapes (n_qp, 8) and (n_qp, 8, 3)
    N_all, dN_dxi_all = shape_functions(xi, eta, zeta)

    # Compute the Jacobian at each quadrature point.
    # J[q, j, k] = sum_{i=0}^{7} element_nodes[i, j] * dN_dxi_all[q, i, k]
    # Shape: (n_qp, 3, 3)
    J = jnp.einsum('ij,qik->qjk', element_nodes, dN_dxi_all)

    # Compute determinant and inverse of the Jacobian.
    detJ = jnp.abs(jnp.linalg.det(J))
    J_inv = jnp.linalg.inv(J)

    # Map the shape function derivatives to physical coordinates:
    # dN_dx[q] = J_inv[q] @ dN_dxi_all[q] for each quadrature point.
    # Shape (n_qp, 8, 3)
    dN_dx = jnp.einsum('qjk,qik->qij', J_inv, dN_dxi_all)

    # For each quadrature point, compute the contribution to the local stiffness:
    # Contribution = k_eff * (dN_dx @ dN_dx^T) * detJ * w_total.
    # Shape: (n_qp, 8, 8)
    contrib = jnp.einsum('qik,qjk->qij', dN_dx, dN_dx)
    contrib = k_eff * contrib * (detJ * w_total)[:, None, None]

    # Sum over all quadrature points to obtain the local stiffness matrix.
    # Shape: (8, 8)
    Ke = jnp.sum(contrib, axis=0)

    return Ke
