import jax.numpy as jnp
from .quadrature import shape_functions


def element_stiffness_matrix(element_nodes, k_eff, gauss_pts, gauss_wts):
    """
    Compute the local 8x8 stiffness matrix for a hexahedral element.

    Parameters:
      element_nodes: (8,3) array of the coordinates of the element's nodes.
      k_eff: effective thermal conductivity for the element (possibly modulated by density).

    Returns:
      Ke: (8,8) element stiffness matrix.
    """
    Ke = jnp.zeros((8, 8))
    # Loop over the 2x2x2 Gauss points with indices to use the weights.
    for i, xi in enumerate(gauss_pts):
        w_xi = gauss_wts[i]
        for j, eta in enumerate(gauss_pts):
            w_eta = gauss_wts[j]
            for k, zeta in enumerate(gauss_pts):
                w_zeta = gauss_wts[k]
                # Total weight for this quadrature point:
                w_total = w_xi * w_eta * w_zeta

                # Evaluate shape functions and their derivatives at (xi, eta, zeta)
                N, dN_dxi = shape_functions(xi, eta, zeta)
                # Compute the Jacobian matrix J (3x3) for this quadrature point.
                J = jnp.zeros((3, 3))
                for i_node in range(8):
                    J = J + jnp.outer(element_nodes[i_node], dN_dxi[i_node])
                detJ = jnp.abs(jnp.linalg.det(J))
                # Map derivatives to physical coordinates: dN_dx = J^{-1} * dN_dxi
                dN_dx = jnp.linalg.solve(J, dN_dxi.T).T  # shape (8,3)
                # Add the weighted contribution to the element stiffness matrix.
                Ke = Ke + k_eff * (dN_dx @ dN_dx.T) * detJ * w_total
    return Ke
