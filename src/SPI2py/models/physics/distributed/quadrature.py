import jax.numpy as jnp


def shape_functions(xi, eta, zeta):
    """
    Vectorized evaluation of the trilinear shape functions and their derivatives.

    Parameters:
      xi, eta, zeta: arrays of shape (n_qp,)

    Returns:
      N: (n_qp, 8) array of shape function values.
      dN_dxi: (n_qp, 8, 3) array of derivatives with respect to (xi,eta,zeta).
    """

    # Reference coordinates for the eight nodes, shape (8, 3)
    node_ref = jnp.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1]
    ])

    # Reshape xi, eta, zeta to shape (n_qp, 1) for broadcasting.
    xi = xi[:, None]
    eta = eta[:, None]
    zeta = zeta[:, None]

    # Evaluate shape functions.
    N = 1/8.0 * (1 + xi * node_ref[:, 0]) * (1 + eta * node_ref[:, 1]) * (1 + zeta * node_ref[:, 2])

    # Compute derivatives with respect to xi, eta, and zeta.
    dN_dxi0 = 1/8.0 * node_ref[:, 0] * (1 + eta * node_ref[:, 1]) * (1 + zeta * node_ref[:, 2])
    dN_dxi1 = 1/8.0 * node_ref[:, 1] * (1 + xi  * node_ref[:, 0]) * (1 + zeta * node_ref[:, 2])
    dN_dxi2 = 1/8.0 * node_ref[:, 2] * (1 + xi  * node_ref[:, 0]) * (1 + eta  * node_ref[:, 1])

    # Broadcast each derivative to shape (n_qp, 8)
    dN_dxi0 = jnp.broadcast_to(dN_dxi0, N.shape)
    dN_dxi1 = jnp.broadcast_to(dN_dxi1, N.shape)
    dN_dxi2 = jnp.broadcast_to(dN_dxi2, N.shape)

    # Stack to get derivatives of shape (n_qp, 8, 3)
    dN_dxi = jnp.stack([dN_dxi0, dN_dxi1, dN_dxi2], axis=-1)

    return N, dN_dxi


def gauss_quad(n_qp=2):
    """
    Return the Gauss quadrature points and weights for a 3D hexahedron.

    Parameters:
      n_qp: Number of quadrature points in each direction.

    Returns:
      gauss_pts: Array of quadrature points of shape (n_qp,).
      gauss_wts: Array of quadrature weights of shape (n_qp,).
    """

    if n_qp == 2:
        # Define Gauss quadrature points (2-point rule in each direction)
        gauss_pts = jnp.array([-1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)])
        gauss_wts = jnp.array([1.0, 1.0])
    else:
        raise NotImplementedError("Only 2-point Gauss quadrature is supported for now.")

    return gauss_pts, gauss_wts
