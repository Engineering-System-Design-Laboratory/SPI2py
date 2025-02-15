import jax.numpy as jnp
from .mesh import generate_mesh_vec, find_active_nodes, find_face_nodes
from .assembly import assemble_global_stiffness_matrix_vec, apply_dirichlet_bc, apply_robin_bc, combine_fixed_conditions


def solve_system(K, f):
    """
    Solve the linear system K*T = f for the nodal temperatures T.
    """
    T = jnp.linalg.solve(K, f)
    return T


# def fea_3d_thermal(nx, ny, nz,
#                    lx, ly, lz,
#                    base_k,
#                    density,
#                    h,
#                    T_inf,
#                    fixed_nodes,
#                    fixed_values,
#                    convection_nodes,
#                    conv_area,
#                    comp_nodes,
#                    comp_temp):
#     """
#     Full pipeline for the 3D FEA:
#       - Mesh generation
#       - Global stiffness assembly (with SIMP density modulation)
#       - Application of convection and Dirichlet boundary conditions
#       - Solving the system for temperature.
#
#     Parameters:
#       nx,ny,nz   : Number of elements in each coordinate direction.
#       lx,ly,lz   : Domain dimensions.
#       base_k     : Base thermal conductivity.
#       density    : (n_elements,) array of density values (from geometry projection).
#       h          : Convection coefficient.
#       T_inf      : Ambient temperature for convection.
#       fixed_nodes: Array of node indices where Dirichlet conditions are applied.
#       fixed_values: Corresponding temperature values at the fixed nodes.
#       convection_nodes: Array of node indices on the convection boundary.
#       conv_area  : The area associated with each convection node.
#
#     Returns:
#       nodes: Nodal coordinates.
#       elements: Element connectivity.
#       T: Computed nodal temperature distribution.
#     """
#     # TODO Apply BC here... Also, capture when multiple BC/loads specified for same node...
#     nodes, elements = generate_mesh_vec(nx, ny, nz, lx, ly, lz)
#
#     #
#     K, f = assemble_global_stiffness_matrix_vec(nodes, elements, density, base_k)
#
#     # Apply boundary conditions.
#     bc_tracer = {}
#     K, f = apply_robin_bc(K, f, convection_nodes, h, T_inf, conv_area)
#     K, f = apply_dirichlet_bc(K, f, fixed_nodes, fixed_values)
#     # f = apply_load(f, comp_nodes, comp_temp)
#     T = solve_system(K, f)
#     return nodes, elements, T


def fea_3d_thermal(nx, ny, nz,
                   lx, ly, lz,
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
    Full pipeline for the 3D FEA:
      - Mesh generation
      - Global stiffness assembly (with SIMP density modulation)
      - Application of convection and Dirichlet boundary conditions
      - Solving the system for temperature.

    Parameters:
      nx,ny,nz   : Number of elements in each coordinate direction.
      lx,ly,lz   : Domain dimensions.
      base_k     : Base thermal conductivity.
      density    : (n_elements,) array of density values (from geometry projection).
      h          : Convection coefficient.
      T_inf      : Ambient temperature for convection.
      fixed_nodes: Array of node indices where Dirichlet conditions are applied.
      fixed_values: Corresponding temperature values at the fixed nodes.
      robin_nodes: Array of node indices on the convection boundary.
      conv_area  : The area associated with each convection node.

    Returns:
      nodes: Nodal coordinates.
      elements: Element connectivity.
      T: Computed nodal temperature distribution.
    """
    # TODO Apply BC here... Also, capture when multiple BC/loads specified for same node...
    nodes, elements = generate_mesh_vec(nx, ny, nz, lx, ly, lz)

    #
    K, f = assemble_global_stiffness_matrix_vec(nodes, elements, density, base_k)

    # Apply boundary conditions.
    bc_tracer = {}
    K, f = apply_robin_bc(K, f, robin_nodes, h, T_inf, conv_area)
    K, f = apply_dirichlet_bc(K, f, fixed_nodes, fixed_values)
    # f = apply_load(f, comp_nodes, comp_temp)
    T = solve_system(K, f)
    return nodes, elements, T


def solve_system_partitioned(nx, ny, nz,
                               lx, ly, lz,
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
    nodes, elements = generate_mesh_vec(nx, ny, nz, lx, ly, lz)
    # Assemble the global stiffness matrix and load vector.
    K, f = assemble_global_stiffness_matrix_vec(nodes, elements, density, base_k)
    n_nodes = K.shape[0]

    # Apply Robin (convective) boundary conditions.
    K_mod, f_mod = apply_robin_bc(K, f, robin_nodes, h, T_inf, conv_area)

    # Partition the system for Dirichlet BCs.
    # (apply_dirichlet_bc_partition is assumed to enforce that fixed_values is already a 1D array.)
    combined_fixed_nodes, combined_fixed_values = combine_fixed_conditions([fixed_nodes, comp_nodes], [fixed_values, comp_temp])
    free_indices, K_ff, f_free = apply_dirichlet_bc(K_mod, f_mod, combined_fixed_nodes, combined_fixed_values)

    # Solve the reduced system for the free degrees of freedom.
    u_free = jnp.linalg.solve(K_ff, f_free)

    # Reassemble the full solution.
    u = jnp.zeros(n_nodes)
    u = u.at[free_indices].set(u_free)


    u = u.at[fixed_nodes].set(fixed_values)

    return nodes, elements, u
