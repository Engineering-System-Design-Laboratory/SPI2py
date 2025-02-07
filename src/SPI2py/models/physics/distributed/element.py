import jax.numpy as jnp
from .quadrature import shape_functions, gauss_pts


def element_stiffness_matrix(element_nodes, k_eff, gauss_pts):
    """
    Compute the local 8x8 stiffness matrix for a hexahedral element.

    Parameters:
      element_nodes: (8,3) array of the coordinates of the element's nodes.
      k_eff: effective thermal conductivity for the element (possibly modulated by density).

    Returns:
      Ke: (8,8) element stiffness matrix.
    """
    Ke = jnp.zeros((8, 8))
    # Loop over the 2x2x2 Gauss points.
    for xi in gauss_pts:
        for eta in gauss_pts:
            for zeta in gauss_pts:
                # Evaluate shape functions and their derivatives at this quadrature point.
                N, dN_dxi = shape_functions(xi, eta, zeta)
                # Compute the Jacobian matrix: J_{mn} = sum_{i=0}^{7} x_i[m] * dN_i/dxi_n
                J = jnp.zeros((3, 3))
                for i in range(8):
                    # Outer product: node coordinate (3,) with derivative (3,)
                    J = J + jnp.outer(element_nodes[i], dN_dxi[i])
                detJ = jnp.abs(jnp.linalg.det(J))
                # Map derivatives to physical coordinates:
                # dN_dx: (8,3) where for each node: dN_dx = J^{-1} * dN_dxi (for that node)
                dN_dx = jnp.linalg.solve(J, dN_dxi.T).T  # shape (8,3)
                # Contribution: for conduction the bilinear form is grad(N_i) dot grad(N_j)
                Ke = Ke + k_eff * (dN_dx @ dN_dx.T) * detJ
    return Ke
