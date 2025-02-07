import jax.numpy as jnp


def generate_mesh(nx, ny, nz, lx, ly, lz):
    """
    Generate a uniform structured grid in 3D (vectorized version).

    Returns:
      nodes: (n_nodes, 3) array of coordinates.
      elements: (n_elements, 8) connectivity array (indices into nodes).
    """
    # Create grid points along each axis.
    x = jnp.linspace(0, lx, nx + 1)
    y = jnp.linspace(0, ly, ny + 1)
    z = jnp.linspace(0, lz, nz + 1)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    nodes = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # The total number of nodes is (nx+1)*(ny+1)*(nz+1).
    # For elements, we want to create connectivity for each cell in a grid of shape (nx, ny, nz).

    # Create arrays of element base indices using meshgrid.
    I, J, K = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing='ij')
    # Flatten to 1D arrays.
    I = I.ravel()
    J = J.ravel()
    K = K.ravel()

    # Each node index in a grid of shape (nx+1, ny+1, nz+1) can be computed as:
    # index = i * ((ny+1) * (nz+1)) + j * (nz+1) + k.
    stride_j = (nz + 1)
    stride_i = (ny + 1) * (nz + 1)

    def idx(i, j, k):
        return i * stride_i + j * stride_j + k

    # For each cell, compute the eight nodes:
    n0 = idx(I,   J,   K)
    n1 = idx(I+1, J,   K)
    n2 = idx(I+1, J+1, K)
    n3 = idx(I,   J+1, K)
    n4 = idx(I,   J,   K+1)
    n5 = idx(I+1, J,   K+1)
    n6 = idx(I+1, J+1, K+1)
    n7 = idx(I,   J+1, K+1)

    # Stack these indices along the last axis. The result has shape (n_elements, 8).
    elements = jnp.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=-1)
    return nodes, elements