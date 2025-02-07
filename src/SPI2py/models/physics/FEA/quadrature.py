import jax.numpy as jnp


def shape_functions(xi, eta, zeta):
    """
    Returns the 8 shape functions and their derivatives with respect to (xi,eta,zeta)
    at a given integration point (xi,eta,zeta) on the reference element.
    """

    # Reference coordinates of nodes in the standard hexahedral element
    node_ref = jnp.array([[-1, -1, -1],
                          [1, -1, -1],
                          [1, 1, -1],
                          [-1, 1, -1],
                          [-1, -1, 1],
                          [1, -1, 1],
                          [1, 1, 1],
                          [-1, 1, 1]])

    # Trilinear shape functions:
    N = 1 / 8.0 * (1 + xi * node_ref[:, 0]) * (1 + eta * node_ref[:, 1]) * (1 + zeta * node_ref[:, 2])

    # Derivatives dN/d(xi,eta,zeta), shape: (8,3)
    dN_dxi = jnp.stack([
        1 / 8.0 * node_ref[:, 0] * (1 + eta * node_ref[:, 1]) * (1 + zeta * node_ref[:, 2]),
        1 / 8.0 * node_ref[:, 1] * (1 + xi * node_ref[:, 0]) * (1 + zeta * node_ref[:, 2]),
        1 / 8.0 * node_ref[:, 2] * (1 + xi * node_ref[:, 0]) * (1 + eta * node_ref[:, 1])
    ], axis=1)

    return N, dN_dxi


# Define Gauss quadrature points (2-point rule in each direction)
gauss_pts = jnp.array([-1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)])
gauss_wts = jnp.array([1.0, 1.0])  # For a 2-point Gauss-Legendre rule in [-1,1]
