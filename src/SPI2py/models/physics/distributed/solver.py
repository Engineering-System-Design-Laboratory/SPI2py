import jax.numpy as jnp
from .mesh import generate_mesh_vec, find_active_nodes, find_face_nodes, generate_mesh
from .assembly import assemble_global_stiffness_matrix, apply_dirichlet_bc, apply_robin_bc, combine_fixed_conditions, partition_global_system



def solve_system_partitioned(nodes, elements,
                               base_k,
                               density,
                               h,
                               T_inf,
                               fixed_nodes,
                               fixed_values,
                               robin_nodes,
                               conv_area,
                               comp_nodes,
                               comp_temp):
    """
    Given a global stiffness matrix K and load vector f, modify them for Robin BCs,
    partition the system for Dirichlet BCs, and solve the reduced system.
    Then reassemble the full solution vector.
    """

    # Generate the mesh and element connectivity.

    # Assemble the global stiffness matrix and load vector.
    K, f = assemble_global_stiffness_matrix(nodes, elements, density, base_k)

    # Apply all boundary conditions and partition the system.


    # Apply Robin (convective) boundary conditions.
    K_mod, f_mod = apply_robin_bc(K, f, robin_nodes, h, T_inf, conv_area)

    # Partition the system for Dirichlet BCs.
    combined_fixed_nodes, combined_fixed_values = combine_fixed_conditions([fixed_nodes, comp_nodes], [fixed_values, comp_temp])
    # prescribed_nodes, prescribed_values = fixed_nodes, fixed_values

    K_ff, K_fp, K_pf, K_pp, D_p, R_f, R_p, idx_f, idx_p = partition_global_system(K_mod, f_mod, combined_fixed_nodes, combined_fixed_values)

    # Solve the partitioned system for the unknown displacements.
    # K_ff D_f + K_fp D_p = R_f
    # K_ff D_f = R_f - K_fp D_p
    D_f = jnp.linalg.solve(K_ff, R_f - K_fp @ D_p)

    # Reassemble the full solution.
    n_nodes = K.shape[0]
    R = jnp.zeros(n_nodes)
    R = R.at[idx_f].set(D_f)
    R = R.at[idx_p].set(D_p)

    return nodes, elements, R
