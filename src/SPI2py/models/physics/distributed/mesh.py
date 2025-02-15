import jax.numpy as jnp


def generate_mesh(nx, ny, nz, lx, ly, lz):
    """
    Generate a uniform structured grid in 3D using explicit Python loops.

    Parameters:
      nx, ny, nz: Number of elements along x, y, z.
      lx, ly, lz: Physical dimensions along x, y, z.

    Returns:
      nodes: (n_nodes, 3) array of coordinates (n_nodes = (nx+1)*(ny+1)*(nz+1)).
      elements: (n_elements, 8) connectivity array (indices into nodes,
                n_elements = nx*ny*nz).
    """

    # Compute spacing
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    # Generate nodes by looping over each coordinate.
    nodes_list = []
    for i in range(nx + 1):
        x = i * dx
        for j in range(ny + 1):
            y = j * dy
            for k in range(nz + 1):
                z = k * dz
                nodes_list.append([x, y, z])
    nodes = jnp.array(nodes_list)

    # Now, create the element connectivity.
    # The node numbering is assumed to be:
    # index = i * ((ny+1) * (nz+1)) + j * (nz+1) + k.
    elements_list = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Compute the indices for the 8 nodes of the hexahedral element.
                n0 = i * ((ny + 1) * (nz + 1)) + j * (nz + 1) + k
                n1 = (i + 1) * ((ny + 1) * (nz + 1)) + j * (nz + 1) + k
                n2 = (i + 1) * ((ny + 1) * (nz + 1)) + (j + 1) * (nz + 1) + k
                n3 = i * ((ny + 1) * (nz + 1)) + (j + 1) * (nz + 1) + k
                n4 = i * ((ny + 1) * (nz + 1)) + j * (nz + 1) + (k + 1)
                n5 = (i + 1) * ((ny + 1) * (nz + 1)) + j * (nz + 1) + (k + 1)
                n6 = (i + 1) * ((ny + 1) * (nz + 1)) + (j + 1) * (nz + 1) + (k + 1)
                n7 = i * ((ny + 1) * (nz + 1)) + (j + 1) * (nz + 1) + (k + 1)
                elements_list.append([n0, n1, n2, n3, n4, n5, n6, n7])
    elements = jnp.array(elements_list)
    return nodes, elements



def generate_mesh_vec(nx, ny, nz, lx, ly, lz):
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


def find_active_nodes(density, threshold=1e-3):
    """
    Given a density array (per node), return the indices of nodes with density above the threshold.
    """
    return jnp.where(density > threshold)[0]


def find_face_nodes(nodes: jnp.ndarray, face_normal: jnp.ndarray, tol: float = 1e-6) -> jnp.ndarray:
    """
    Given an array of nodes (shape: [n_nodes, 3]) and a face normal (e.g., [0, 1, 0] for the top),
    return the indices of nodes on the face corresponding to the maximum projection in that direction.

    Parameters:
      nodes: (n_nodes, 3) array of node coordinates.
      face_normal: A 3-element array indicating the direction of the face normal.
                   For example, [0, 1, 0] for the top face.
      tol: Tolerance for deciding if a node is on the face.

    Returns:
      A 1D array of indices corresponding to nodes on that face.
    """
    # Normalize the face normal.
    n_unit = face_normal / jnp.linalg.norm(face_normal)

    # Compute dot products between each node and the face normal.
    dots = nodes @ n_unit  # shape: (n_nodes,)

    # The face we want is the one with the maximum projection.
    max_dot = jnp.max(dots)

    # Select nodes that are within 'tol' of the maximum dot product.
    face_indices = jnp.where(max_dot - dots < tol)[0]
    return face_indices

