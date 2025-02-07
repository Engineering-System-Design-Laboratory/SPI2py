import jax
import jax.numpy as jnp
from .quadrature import shape_functions


# def element_stiffness_matrix(element_nodes, k_eff, gauss_pts, gauss_wts):
#     """
#     Compute the local 8x8 stiffness matrix for a hexahedral element.
#
#     Parameters:
#       element_nodes: (8,3) array of the coordinates of the element's nodes.
#       k_eff: effective thermal conductivity for the element (possibly modulated by density).
#
#     Returns:
#       Ke: (8,8) element stiffness matrix.
#     """
#     Ke = jnp.zeros((8, 8))
#     # Loop over the 2x2x2 Gauss points with indices to use the weights.
#     for i, xi in enumerate(gauss_pts):
#         w_xi = gauss_wts[i]
#         for j, eta in enumerate(gauss_pts):
#             w_eta = gauss_wts[j]
#             for k, zeta in enumerate(gauss_pts):
#                 w_zeta = gauss_wts[k]
#                 # Total weight for this quadrature point:
#                 w_total = w_xi * w_eta * w_zeta
#
#                 # Evaluate shape functions and their derivatives at (xi, eta, zeta)
#                 N, dN_dxi = shape_functions(xi, eta, zeta)
#                 # Compute the Jacobian matrix J (3x3) for this quadrature point.
#                 J = jnp.zeros((3, 3))
#                 for i_node in range(8):
#                     J = J + jnp.outer(element_nodes[i_node], dN_dxi[i_node])
#                 detJ = jnp.abs(jnp.linalg.det(J))
#                 # Map derivatives to physical coordinates: dN_dx = J^{-1} * dN_dxi
#                 dN_dx = jnp.linalg.solve(J, dN_dxi.T).T  # shape (8,3)
#                 # Add the weighted contribution to the element stiffness matrix.
#                 Ke = Ke + k_eff * (dN_dx @ dN_dx.T) * detJ * w_total
#     return Ke

def element_stiffness_matrix(element_nodes, k_eff, gauss_pts, gauss_wts):
    """
    Compute the 8x8 stiffness matrix for a hexahedral element using vectorized quadrature.

    Parameters:
      element_nodes: (8,3) array of node coordinates.
      k_eff        : effective thermal conductivity for the element.
      gauss_pts    : 1D array of Gauss quadrature points (e.g. jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])).
      gauss_wts    : 1D array of Gauss quadrature weights (e.g. jnp.array([1.0, 1.0])).

    Returns:
      Ke: (8,8) element stiffness matrix.
    """

    # Build a tensor grid of quadrature points in 3D.
    # This produces arrays of shape (n, n, n) where n = len(gauss_pts).
    xi_grid, eta_grid, zeta_grid = jnp.meshgrid(gauss_pts, gauss_pts, gauss_pts, indexing='ij')

    # Flatten to shape (nqp,), where nqp = n^3.
    xi = xi_grid.flatten()
    eta = eta_grid.flatten()
    zeta = zeta_grid.flatten()

    # Build corresponding weights.
    wx, wy, wz = jnp.meshgrid(gauss_wts, gauss_wts, gauss_wts, indexing='ij')
    w_total = (wx * wy * wz).flatten()  # shape (nqp,)

    # Vectorize shape function evaluation.
    # Assume shape_functions(xi,eta,zeta) returns:
    #   N: (8,) array of shape functions
    #   dN_dxi: (8, 3) array of derivatives with respect to (xi,eta,zeta)
    v_shape_functions = jax.vmap(lambda a, b, c: shape_functions(a, b, c))
    N_all, dN_dxi_all = v_shape_functions(xi, eta, zeta)
    # N_all has shape (nqp, 8)
    # dN_dxi_all has shape (nqp, 8, 3)

    # For each quadrature point, compute the Jacobian:
    # J = sum_{i=0}^{7} outer(element_nodes[i], dN_dxi[i])
    # This is equivalent to: J = element_nodes.T @ dN_dxi (for each quadrature point).
    v_compute_J = jax.vmap(lambda dN: element_nodes.T @ dN)
    J_all = v_compute_J(dN_dxi_all)  # shape (nqp, 3, 3)

    # Compute determinants of the Jacobians.
    detJ_all = jnp.abs(jnp.linalg.det(J_all))  # shape (nqp,)

    # For each quadrature point, map dN/dxi to physical space:
    # dN_dx = J^{-1} * dN_dxi, which we compute for each quadrature point.
    v_solve = jax.vmap(lambda J, dN: jnp.linalg.solve(J, dN.T).T)
    dN_dx_all = v_solve(J_all, dN_dxi_all)  # shape (nqp, 8, 3)

    # For each quadrature point, compute the local contribution:
    # Contribution = k_eff * (dN_dx @ dN_dx.T) * detJ * (weight)
    v_contrib = jax.vmap(lambda dNdx, detJ, wt: k_eff * (dNdx @ dNdx.T) * detJ * wt)
    contributions = v_contrib(dN_dx_all, detJ_all, w_total)  # shape (nqp, 8, 8)

    # Sum contributions from all quadrature points.
    Ke = jnp.sum(contributions, axis=0)  # shape (8, 8)
    return Ke