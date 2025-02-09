import jax.numpy as jnp
from .element import element_stiffness_matrix
from .quadrature import gauss_pts, gauss_wts


def assemble_global_system(nodes, elements, density, base_k):
    """
    Assemble the global stiffness matrix K and load vector f.

    Parameters:
      nodes: (n_nodes,3) array of coordinates.
      elements: (n_elements,8) connectivity array.
      density: (n_elements,) array representing material densities (from projection).
      base_k: base thermal conductivity.

    Returns:
      K_global: (n_nodes,n_nodes) global stiffness matrix.
      f_global: (n_nodes,) load vector (zero in this simple example).
    """
    n_nodes = nodes.shape[0]
    K_global = jnp.zeros((n_nodes, n_nodes))
    f_global = jnp.zeros(n_nodes)

    n_elem = elements.shape[0]
    for e in range(n_elem):

        # Effective conductivity
        k_eff = base_k * density[e]
        elem_nodes_idx = elements[e]
        elem_nodes = nodes[elem_nodes_idx]
        Ke = element_stiffness_matrix(elem_nodes, k_eff, gauss_pts, gauss_wts)  # TODO and weights?

        # Scatter assembly into the global stiffness matrix:
        for i_local in range(8):
            i_global = elem_nodes_idx[i_local]
            for j_local in range(8):
                j_global = elem_nodes_idx[j_local]
                K_global = K_global.at[i_global, j_global].add(Ke[i_local, j_local])

    return K_global, f_global


def apply_dirichlet_bc(K, f, fixed_indices, fixed_value, penalty=1e12):
    """
    Enforce Dirichlet (fixed) boundary conditions on the DOFs in fixed_indices.
    This sets the corresponding rows in K to zero, then sets the diagonal to a large number (penalty),
    and adjusts f to enforce the fixed value.

    Parameters:
      K: Global stiffness matrix (n_nodes x n_nodes).
      f: Global load vector (n_nodes,).
      fixed_indices: 1D array of node indices (DOFs) to be fixed.
      fixed_value: The prescribed value at these DOFs.
      penalty: A large number used to enforce the condition.

    Returns:
      Updated (K, f) with Dirichlet conditions applied.
    """
    # Zero out the rows corresponding to fixed indices.
    K = K.at[fixed_indices, :].set(0.0)

    # Set the diagonal entries for these indices to a large penalty.
    K = K.at[fixed_indices, fixed_indices].set(penalty)

    # Adjust the load vector so that f[i] = penalty * fixed_value for each fixed DOF.
    f = f.at[fixed_indices].set(penalty * fixed_value)

    return K, f


def apply_robin_bc(K, f, robin_indices, h, Tinf, area):
    """
    Enforce Robin (convective) boundary conditions on the DOFs in robin_indices.
    This adds h*area to the diagonal of K and h*area*Tinf to f.

    Parameters:
      K: Global stiffness matrix (n_nodes x n_nodes).
      f: Global load vector (n_nodes,).
      robin_indices: 1D array of node indices to which the Robin BC is applied.
      h: Convection coefficient.
      Tinf: Ambient temperature.
      area: The effective area associated with each node.

    Returns:
      Updated (K, f) with Robin BC contributions.
    """
    # Add the convective contribution to the diagonal of K.
    K = K.at[robin_indices, robin_indices].add(h * area)
    # Add the corresponding contribution to f.
    f = f.at[robin_indices].add(h * area * Tinf)
    return K, f


def apply_load(f, load_indices, load_value):
    """
    Apply a prescribed load to the specified nodes by adding load_value to f.

    Parameters:
      f: Global load vector (n_nodes,).
      load_indices: 1D array of node indices where the load is applied.
      load_value: The load value (e.g., a heat source or force).

    Returns:
      Updated load vector f.
    """
    f = f.at[load_indices].add(load_value)
    return f

