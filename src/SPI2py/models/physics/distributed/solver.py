import jax.numpy as jnp
from .mesh import generate_mesh, find_active_nodes, find_face_nodes
from .assembly import assemble_global_system, apply_dirichlet_bc, apply_robin_bc, apply_load


def solve_system(K, f):
    """
    Solve the linear system K*T = f for the nodal temperatures T.
    """
    T = jnp.linalg.solve(K, f)
    return T


def fea_3d_thermal(nx, ny, nz,
                   lx, ly, lz,
                   base_k,
                   density,
                   h,
                   T_inf,
                   fixed_nodes,
                   fixed_values,
                   convection_nodes,
                   conv_area):
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
      convection_nodes: Array of node indices on the convection boundary.
      conv_area  : The area associated with each convection node.

    Returns:
      nodes: Nodal coordinates.
      elements: Element connectivity.
      T: Computed nodal temperature distribution.
    """
    # TODO Apply BC here... Also, capture when multiple BC/loads specified for same node...
    nodes, elements = generate_mesh(nx, ny, nz, lx, ly, lz)
    K, f = assemble_global_system(nodes, elements, density, base_k)
    K, f = apply_robin_bc(K, f, convection_nodes, h, T_inf, conv_area)
    K, f = apply_dirichlet_bc(K, f, fixed_nodes, fixed_values)
    T = solve_system(K, f)
    return nodes, elements, T

