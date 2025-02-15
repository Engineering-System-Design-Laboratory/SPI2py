import jax
import jax.numpy as jnp
from .element import assemble_local_stiffness_matrix
from .quadrature import shape_functions_vec, gauss_quad


def assemble_global_stiffness_matrix(nodes, elements, density, base_k, gauss_pts, gauss_wts):
    n_nodes = nodes.shape[0]
    K_global = jnp.zeros((n_nodes, n_nodes))
    n_elem = elements.shape[0]
    for e in range(n_elem):
        k_eff = base_k * density[e]
        elem_nodes_idx = elements[e]
        elem_nodes = nodes[elem_nodes_idx]
        Ke = assemble_local_stiffness_matrix(elem_nodes, k_eff, gauss_pts, gauss_wts)
        for i_local in range(8):
            i_global = elem_nodes_idx[i_local]
            for j_local in range(8):
                j_global = elem_nodes_idx[j_local]
                K_global = K_global.at[i_global, j_global].add(Ke[i_local, j_local])
    return K_global


def assemble_global_stiffness_matrix_vec(nodes, elements, density, base_k):
    """
    Assemble the global stiffness matrix K and load vector f in a fully vectorized way without using vmap.

    Parameters:
      nodes:      (n_nodes, 3) array of coordinates.
      elements:   (n_elem, 8) connectivity array (indices into nodes).
      density:    (n_elem,) array of material densities.
      base_k:     Base thermal conductivity.
      gauss_pts:  1D array of Gauss quadrature points.
      gauss_wts:  1D array of Gauss quadrature weights.

    Returns:
      K_global:   (n_nodes, n_nodes) global stiffness matrix.
      f_global:   (n_nodes,) load vector (zero in this example).
    """
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]

    # 1. Compute effective conductivity for each element.
    #    (Assume a simple linear interpolation: k_eff = base_k * density)
    k_eff_all = base_k * density  # shape: (n_elem,)

    # 2. Gather nodal coordinates for all elements.
    #    element_nodes_all will have shape (n_elem, 8, 3)
    element_nodes_all = nodes[elements]

    # 3. Build the quadrature grid for the element integration.
    #    Create a tensor grid from the 1D gauss_pts and gauss_wts.
    gauss_pts, gauss_wts = gauss_quad(n_qp=2)
    xi_grid, eta_grid, zeta_grid = jnp.meshgrid(gauss_pts, gauss_pts, gauss_pts, indexing='ij')
    xi = xi_grid.flatten()  # shape: (n_qp,)
    eta = eta_grid.flatten()  # shape: (n_qp,)
    zeta = zeta_grid.flatten()  # shape: (n_qp,)
    n_qp = xi.shape[0]

    # 4. Build the corresponding quadrature weights.
    wx, wy, wz = jnp.meshgrid(gauss_wts, gauss_wts, gauss_wts, indexing='ij')
    w_total = (wx * wy * wz).flatten()  # shape: (n_qp,)

    # 5. Evaluate the shape functions and their natural derivatives at all quadrature points.
    #    shape_functions_vec should accept arrays of quadrature points and return:
    #      N_all: (n_qp, 8)
    #      dN_dxi_all: (n_qp, 8, 3)
    N_all, dN_dxi_all = shape_functions_vec(xi, eta, zeta)

    # 6. Compute the Jacobian for each element at each quadrature point.
    #    For each element e and quadrature point q:
    #       J_all[e, q, j, k] = sum_{i=0}^{7} element_nodes_all[e, i, j] * dN_dxi_all[q, i, k]
    #    This can be done with einsum:
    J_all = jnp.einsum('eij,qik->eqjk', element_nodes_all, dN_dxi_all)
    # J_all has shape: (n_elem, n_qp, 3, 3)

    # 7. Compute the determinant and inverse of each Jacobian.
    detJ_all = jnp.abs(jnp.linalg.det(J_all))  # shape: (n_elem, n_qp)
    J_inv_all = jnp.linalg.inv(J_all)  # shape: (n_elem, n_qp, 3, 3)

    # 8. Map the shape function derivatives to physical coordinates.
    #    For each element and quadrature point:
    #       dN_dx = J_inv * dN_dxi.
    #    First, broadcast dN_dxi_all from shape (n_qp, 8, 3) to (n_elem, n_qp, 8, 3):
    dN_dxi_all_b = jnp.broadcast_to(dN_dxi_all, (n_elem, n_qp, 8, 3))
    # Now compute dN_dx_all with einsum:
    dN_dx_all = jnp.einsum('eqjk,eqik->eqij', J_inv_all, dN_dxi_all_b)
    # dN_dx_all has shape: (n_elem, n_qp, 8, 3)

    # 9. Compute the element stiffness contributions for each element and quadrature point.
    #    For each element e and quadrature point q, the contribution is:
    #      contrib[e, q] = k_eff_all[e] * (dN_dx_all[e,q] @ dN_dx_all[e,q]^T) * detJ_all[e,q] * w_total[q]
    contrib_all = jnp.einsum('eqik,eqjk->eqij', dN_dx_all, dN_dx_all)  # shape: (n_elem, n_qp, 8, 8)
    contrib_all = k_eff_all[:, None, None, None] * contrib_all \
                  * detJ_all[:, :, None, None] * w_total[None, :, None, None]

    # 10. Sum contributions over quadrature points for each element to obtain element stiffness matrices.
    Ke_all = jnp.sum(contrib_all, axis=1)  # shape: (n_elem, 8, 8)

    # 11. Assemble the global stiffness matrix via vectorized scatter-add.
    #     For each element, we need to add its 8x8 block to the global K.
    #     Build global index arrays for rows and columns:
    #       rows: for each element, repeat its 8 node indices 8 times.
    #       cols: for each element, tile its 8 node indices 8 times.
    rows = jnp.repeat(elements, repeats=8, axis=1)  # shape: (n_elem, 64)
    cols = jnp.tile(elements, reps=(1, 8))  # shape: (n_elem, 64)

    # Flatten these index arrays.
    rows_flat = rows.reshape(-1)  # shape: (n_elem*64,)
    cols_flat = cols.reshape(-1)

    # Flatten the element stiffness matrices.
    Ke_flat = Ke_all.reshape(-1)

    # Initialize the global stiffness matrix and scatter-add contributions.
    K_global = jnp.zeros((n_nodes, n_nodes))
    K_global = K_global.at[rows_flat, cols_flat].add(Ke_flat)

    # For this simple example, the load vector is zero.
    f_global = jnp.zeros(n_nodes)
    return K_global, f_global


def combine_fixed_conditions(fixed_nodes_list, fixed_values_list):
    """
    Combine multiple sets of fixed nodes and their prescribed values into single arrays.

    Parameters:
      fixed_nodes_list: a list (or tuple) of 1D arrays of fixed node indices.
      fixed_values_list: a list (or tuple) of 1D arrays (or scalars) of prescribed values,
                         corresponding to each set of fixed nodes.

    Returns:
      combined_fixed_nodes: a 1D array containing all fixed node indices.
      combined_fixed_values: a 1D array containing the prescribed value for each fixed node.

    Note: If any of the fixed_values in the list is a scalar, it is broadcasted to match the size
    of its corresponding fixed_nodes array.
    """
    combined_nodes = []
    combined_values = []
    for nodes_i, values_i in zip(fixed_nodes_list, fixed_values_list):
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

    # Broadcast fixed_values to have shape (number of fixed DOFs,)
    fixed_values = jnp.broadcast_to(fixed_values, (fixed_indices.shape[0],))

    n_nodes = K.shape[0]
    all_indices = jnp.arange(n_nodes)
    free_indices = jnp.setdiff1d(all_indices, fixed_indices)

    K_ff = K[free_indices][:, free_indices]
    K_fd = K[free_indices][:, fixed_indices]
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
    K_new = K.at[robin_indices, robin_indices].add(h * area)
    # Add the corresponding contribution to the load vector.
    f_new = f.at[robin_indices].add(h * area * T_inf)
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

