import jax
from jax import vmap
import jax.numpy as jnp
from .element import assemble_local_stiffness_matrix
from .quadrature import gauss_quad


def assemble_global_stiffness_matrix(nodes, elements, density, base_k):
    """
    Assemble the global stiffness matrix K and load vector f in a fully vectorized way.

    Parameters:
      nodes:    (n_nodes, 3) array of coordinates.
      elements: (n_elem, 8) connectivity array (indices into nodes).
      density:  (n_elem,) array of material densities.
      base_k:   Base conductivity.

    Returns:
      K_global: (n_nodes, n_nodes) global stiffness matrix.
      f_global: (n_nodes,) load vector.
    """
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]

    # Compute effective conductivity for each element.
    # Shape: (n_elem,)
    k_eff_all = base_k * density

    # Gather nodal coordinates for all elements.
    # Shape: (n_elem, 8, 3)
    element_nodes_all = nodes[elements]

    # Get Gauss quadrature points and weights.
    gauss_pts, gauss_wts = gauss_quad(n_qp=2)

    # Use vmap to compute the local stiffness matrix for each element.
    # Shape: (n_elem, 8, 8)
    def compute_local_stiffness_matrix(el_nodes, k_eff):
        return assemble_local_stiffness_matrix(el_nodes, k_eff, gauss_pts, gauss_wts)

    Ke_all = vmap(compute_local_stiffness_matrix)(element_nodes_all, k_eff_all)

    # Assemble the global stiffness matrix via a vectorized scatter-add.
    # For each element, create index arrays for its 8x8 block.
    # Shape: (n_elem, 64)
    rows = jnp.repeat(elements, repeats=8, axis=1)
    cols = jnp.tile(elements, reps=(1, 8))

    rows_flat = rows.reshape(-1)
    cols_flat = cols.reshape(-1)
    Ke_flat = Ke_all.reshape(-1)

    # Initialize and assemble the global stiffness matrix.
    K_global = jnp.zeros((n_nodes, n_nodes))
    K_global = K_global.at[rows_flat, cols_flat].add(Ke_flat)

    # Initialize the global load vector.
    f_global = jnp.zeros(n_nodes)

    return K_global, f_global


def append_global_system(K, f, append_indices, K_add, f_add):
    """
    Modify the global stiffness matrix K and load vector f by updating values at specified nodes.

    Parameters:
      K: Global stiffness matrix (n_nodes x n_nodes).
      f: Global load vector (n_nodes,).
      append_indices: 1D array of node indices to be modified.
      K_add: Stiffness matrix to add at these nodes.
      f_add: Load vector to add at these nodes.

    Returns:
      K_new: Modified stiffness matrix.
      f_new: Modified load vector.
    """
    K_new = K.at[append_indices, append_indices].add(K_add)
    f_new = f.at[append_indices].add(f_add)
    return K_new, f_new


def partition_global_system(K, f, prescribed_indices, prescribed_values):
    """
    Set the values of the global stiffness matrix K and load vector f at specified nodes.

    Parameters:
      K: Global stiffness matrix (n_nodes x n_nodes).
      f: Global load vector (n_nodes,).
      prescribed_indices: 1D array of node indices to be prescribed.
      prescribed_values: 1D array of prescribed values at these nodes.

    Returns:
      K_new: Modified stiffness matrix.
      f_new: Modified load vector.
    """

    # Obtain the number of nodes and all node indices.
    n_nodes = K.shape[0]
    all_indices = jnp.arange(n_nodes)

    # Find the free indices by subtracting the fixed indices from all indices.
    idx_f = jnp.setdiff1d(all_indices, prescribed_indices)
    idx_p = prescribed_indices

    # Partition the stiffness matrix and load vector.
    # [K_ff K_fp] {D_f} = {R_f}
    # [K_pf K_pp] {D_p} = {R_p}
    K_ff = K[idx_f][:, idx_f]
    K_fp = K[idx_f][:, prescribed_indices]
    K_pf = K[prescribed_indices][:, idx_f]
    K_pp = K[prescribed_indices][:, prescribed_indices]
    D_p = prescribed_values
    R_f = f[idx_f]
    R_p = f[prescribed_indices]

    return K_ff, K_fp, K_pf, K_pp, D_p, R_f, R_p, idx_f, idx_p


def combine_fixed_conditions(idx_p, D_p):
    """
    Combine multiple sets of fixed nodes and their prescribed values into single arrays.

    Parameters:
      idx_p: a list (or tuple) of 1D arrays of prescribed node indices.
      D_p: a list (or tuple) of 1D arrays (or scalars) of prescribed values,
                         corresponding to each set of fixed nodes.

    Returns:
      combined_fixed_nodes: a 1D array containing all fixed node indices.
      combined_fixed_values: a 1D array containing the prescribed value for each fixed node.
    """
    combined_nodes = []
    combined_values = []
    for nodes_i, values_i in zip(idx_p, D_p):
        # Ensure values_i is a 1D array broadcasted to the same length as nodes_i.
        values_i = jnp.broadcast_to(jnp.atleast_1d(values_i), (nodes_i.shape[0],))
        combined_nodes.append(nodes_i)
        combined_values.append(values_i)

    combined_fixed_nodes = jnp.concatenate(combined_nodes)
    combined_fixed_values = jnp.concatenate(combined_values)
    return combined_fixed_nodes, combined_fixed_values


def apply_dirichlet_bc(K, f, fixed_indices, fixed_values):
    """
    Partition the global system to enforce Dirichlet (fixed) BCs.

    Parameters:
      K: Global stiffness matrix (n_nodes x n_nodes).
      f: Global load vector (n_nodes,).
      fixed_indices: 1D array of node indices (DOFs) to be fixed.
      fixed_values: 1D array (or scalar broadcastable) of prescribed values at these DOFs.

    Returns:
      free_indices: 1D array of indices corresponding to free DOFs.
      K_ff: Reduced stiffness matrix for free DOFs.
      f_free: Modified load vector for free DOFs: f_free = f_free - K_fd * fixed_values.
    """

    # Obtain the number of nodes and all node indices.
    n_nodes = K.shape[0]
    all_indices = jnp.arange(n_nodes)

    # Find the free indices by subtracting the fixed indices from all indices.
    free_indices = jnp.setdiff1d(all_indices, fixed_indices)

    # Partition the stiffness matrix and load vector.
    K_ff = K[free_indices][:, free_indices]
    K_fd = K[free_indices][:, fixed_indices]

    # Compute the modified load vector for free DOFs.
    # K_ff u_f + K_fp u_p = f_f
    # K_ff u_f = f_f - K_fp u_p
    f_free = f[free_indices] - K_fd @ fixed_values

    return free_indices, K_ff, f_free


def apply_robin_bc(K, f, robin_indices, h, T_inf, area):
    """
    Incorporate a Robin (convective) boundary condition by modifying K and f.

    For nodes in robin_indices, add h*area to the diagonal of K and
    add h*area*T_inf to f.

    Parameters:
      K: Global stiffness matrix (n_nodes x n_nodes).
      f: Global load vector (n_nodes,).
      robin_indices: 1D array of node indices where the Robin BC is applied.
      h: Convection coefficient.
      T_inf: Ambient (free-stream) temperature.
      area: Effective area associated with each Robin node.

    Returns:
      K_new, f_new: The modified stiffness matrix and load vector.
    """
    # Add the Robin (convective) contribution to the diagonal entries.
    K_add = (h * area)

    # Add the corresponding contribution to the load vector.
    f_add = (h * area * T_inf)

    K_new, f_new = append_global_system(K, f, robin_indices, K_add, f_add)

    return K_new, f_new


# def apply_load(f, load_indices, load_value):
#     """
#     Apply a prescribed load to the specified nodes by adding load_value to f.
#
#     Parameters:
#       f: Global load vector (n_nodes,).
#       load_indices: 1D array of node indices where the load is applied.
#       load_value: The load value (e.g., a heat source or force).
#
#     Returns:
#       Updated load vector f.
#
#     """
#     f = f.at[load_indices].add(load_value)
#     return f





# def combine_fixed_conditions(fixed_nodes_list, fixed_values_list):
#     """
#     Combine multiple sets of fixed nodes and their prescribed values into single arrays.
#
#     Parameters:
#       fixed_nodes_list: a list (or tuple) of 1D arrays of fixed node indices.
#       fixed_values_list: a list (or tuple) of 1D arrays (or scalars) of prescribed values,
#                          corresponding to each set of fixed nodes.
#
#     Returns:
#       combined_fixed_nodes: a 1D array containing all fixed node indices.
#       combined_fixed_values: a 1D array containing the prescribed value for each fixed node.
#
#     Note: If any of the fixed_values in the list is a scalar, it is broadcasted to match the size
#     of its corresponding fixed_nodes array.
#     """
#     combined_nodes = []
#     combined_values = []
#     for nodes_i, values_i in zip(fixed_nodes_list, fixed_values_list):
#         # Ensure values_i is a 1D array broadcasted to the same length as nodes_i.
#         values_i = jnp.broadcast_to(jnp.atleast_1d(values_i), (nodes_i.shape[0],))
#         combined_nodes.append(nodes_i)
#         combined_values.append(values_i)
#
#     combined_fixed_nodes = jnp.concatenate(combined_nodes)
#     combined_fixed_values = jnp.concatenate(combined_values)
#     return combined_fixed_nodes, combined_fixed_values
