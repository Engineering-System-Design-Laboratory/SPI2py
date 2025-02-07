import jax.numpy as jnp
from .element import element_stiffness_matrix
from .quadrature import gauss_pts


def assemble_global_system(nodes, elements, density, base_k, penal):
    """
    Assemble the global stiffness matrix K and load vector f.

    Parameters:
      nodes: (n_nodes,3) array of coordinates.
      elements: (n_elements,8) connectivity array.
      density: (n_elements,) array representing material densities (from projection).
      base_k: base thermal conductivity.
      penal: penalization exponent.

    Returns:
      K_global: (n_nodes,n_nodes) global stiffness matrix.
      f_global: (n_nodes,) load vector (zero in this simple example).
    """
    n_nodes = nodes.shape[0]
    K_global = jnp.zeros((n_nodes, n_nodes))
    f_global = jnp.zeros(n_nodes)

    n_elem = elements.shape[0]
    for e in range(n_elem):
        # Effective conductivity; if density is nearly binary from projection,
        # you may choose penal=1 or a reduced exponent.
        k_eff = base_k * (density[e] ** penal)
        elem_nodes_idx = elements[e]
        elem_nodes = nodes[elem_nodes_idx]
        Ke = element_stiffness_matrix(elem_nodes, k_eff, gauss_pts)  # TODO and weights?
        # Scatter assembly into the global stiffness matrix:
        for i_local in range(8):
            i_global = elem_nodes_idx[i_local]
            for j_local in range(8):
                j_global = elem_nodes_idx[j_local]
                K_global = K_global.at[i_global, j_global].add(Ke[i_local, j_local])
    return K_global, f_global


def apply_dirichlet_bc(K, f, fixed_nodes, fixed_values, penalty=1e12):
    """
    Enforce Dirichlet (essential) boundary conditions by a penalty approach.

    For each fixed node, the corresponding row/diagonal of K is set with a large value,
    and f is set to enforce the fixed temperature.
    """
    for idx, value in zip(fixed_nodes, fixed_values):
        K = K.at[idx, idx].set(penalty)
        f = f.at[idx].set(penalty * value)
    return K, f


def apply_convection_bc(K, f, convection_nodes, h, T_inf, conv_area):
    """
    Apply a Robin (convection) condition on the convection boundary.

    For each convection node, add h*conv_area to the diagonal of K and
    h*conv_area*T_inf to f.
    """
    for idx in convection_nodes:
        K = K.at[idx, idx].add(h * conv_area)
        f = f.at[idx].add(h * conv_area * T_inf)
    return K, f