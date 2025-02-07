import jax.numpy as jnp


def generate_mesh(nx, ny, nz, lx, ly, lz):
    """
    Generate a uniform structured grid in 3D.

    Returns:
      nodes: (n_nodes,3) array of coordinates.
      elements: (n_elements,8) connectivity array (indices into nodes).
    """
    x = jnp.linspace(0, lx, nx + 1)
    y = jnp.linspace(0, ly, ny + 1)
    z = jnp.linspace(0, lz, nz + 1)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    nodes = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    elem_list = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Calculate the node indices for the current element
                n0 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
                n1 = (i + 1) * (ny + 1) * (nz + 1) + j * (nz + 1) + k
                n2 = (i + 1) * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + k
                n3 = i * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + k
                n4 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + (k + 1)
                n5 = (i + 1) * (ny + 1) * (nz + 1) + j * (nz + 1) + (k + 1)
                n6 = (i + 1) * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + (k + 1)
                n7 = i * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + (k + 1)
                elem_list.append([n0, n1, n2, n3, n4, n5, n6, n7])
    elements = jnp.array(elem_list)
    return nodes, elements