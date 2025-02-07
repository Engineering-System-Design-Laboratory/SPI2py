import numpy as np
import pyvista as pv
import jax
import jax.numpy as jnp


def plot_temperature_distribution_pyvista(nodes, T, convection_nodes, fixed_nodes,
                                          dims=None, origin=(0, 0, 0), cmap="inferno", opacity=0.75):
    """
    Visualize the 3D temperature distribution on a structured grid using PyVista.

    Parameters:
      nodes            : NumPy array of shape (n_nodes,3) with node coordinates.
      T                : NumPy array of nodal temperatures.
      convection_nodes : 1D NumPy array of indices for nodes on the convection boundary.
      fixed_nodes      : 1D NumPy array of indices for nodes with Dirichlet conditions.
      dims             : Tuple (nx+1, ny+1, nz+1) defining grid dimensions. If None, it is inferred.
      origin           : Grid origin (default (0,0,0)).
      cmap             : Colormap for temperature (default "inferno").
      opacity          : Mesh opacity.
    """
    # Infer grid dimensions if not provided.
    if dims is None:
        unique_x = np.unique(nodes[:, 0])
        unique_y = np.unique(nodes[:, 1])
        unique_z = np.unique(nodes[:, 2])
        dims = (len(unique_x), len(unique_y), len(unique_z))

    # Reshape nodes into (nx+1, ny+1, nz+1, 3) array.
    try:
        grid_points = nodes.reshape(dims + (3,))
    except ValueError:
        raise ValueError("Nodes cannot be reshaped to the provided dimensions.")

    # Create the StructuredGrid by setting points (flattened) and dimensions.
    grid = pv.StructuredGrid()
    grid.points = grid_points.reshape(-1, 3)
    grid.dimensions = dims
    grid["Temperature"] = T

    # Create PolyData for convection boundary and fixed (Dirichlet) nodes.
    conv_points = nodes[convection_nodes]
    fixed_points = nodes[fixed_nodes]
    conv_poly = pv.PolyData(conv_points)
    fixed_poly = pv.PolyData(fixed_points)

    # Setup PyVista plotter.
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars="Temperature", cmap=cmap, opacity=opacity, show_edges=True)
    plotter.add_mesh(conv_poly, color="blue", point_size=10, render_points_as_spheres=True, label="Convection BC")
    plotter.add_mesh(fixed_poly, color="red", point_size=10, render_points_as_spheres=True, label="Dirichlet BC")
    plotter.add_legend()
    plotter.show()


# -------------------------------
# 1. Define Gauss quadrature points (2-point rule in each direction)
# -------------------------------
gauss_pts = jnp.array([-1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)])
gauss_wts = jnp.array([1.0, 1.0])  # For a 2-point Gauss-Legendre rule in [-1,1]


# -------------------------------
# 2. Shape functions for an 8-node hexahedral element
# -------------------------------
def shape_functions(xi, eta, zeta):
    """
    Returns the 8 shape functions and their derivatives with respect to (xi,eta,zeta)
    at a given integration point (xi,eta,zeta) on the reference element.
    """
    # Reference coordinates of nodes in the standard hexahedral element
    node_ref = jnp.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ])
    # Trilinear shape functions:
    N = 1 / 8.0 * (1 + xi * node_ref[:, 0]) * (1 + eta * node_ref[:, 1]) * (1 + zeta * node_ref[:, 2])
    # Derivatives dN/d(xi,eta,zeta)
    dN_dxi = jnp.stack([
        1 / 8.0 * node_ref[:, 0] * (1 + eta * node_ref[:, 1]) * (1 + zeta * node_ref[:, 2]),
        1 / 8.0 * node_ref[:, 1] * (1 + xi * node_ref[:, 0]) * (1 + zeta * node_ref[:, 2]),
        1 / 8.0 * node_ref[:, 2] * (1 + xi * node_ref[:, 0]) * (1 + eta * node_ref[:, 1])
    ], axis=1)  # Shape: (8,3)
    return N, dN_dxi


# -------------------------------
# 3. Compute the element stiffness matrix
# -------------------------------
def element_stiffness_matrix(element_nodes, k_eff):
    """
    Compute the local 8x8 stiffness matrix for a hexahedral element.

    Parameters:
      element_nodes: (8,3) array of the coordinates of the element's nodes.
      k_eff: effective thermal conductivity for the element (possibly modulated by density).

    Returns:
      Ke: (8,8) element stiffness matrix.
    """
    Ke = jnp.zeros((8, 8))
    # Loop over the 2x2x2 Gauss points.
    for xi in gauss_pts:
        for eta in gauss_pts:
            for zeta in gauss_pts:
                # Evaluate shape functions and their derivatives at this quadrature point.
                N, dN_dxi = shape_functions(xi, eta, zeta)
                # Compute the Jacobian matrix: J_{mn} = sum_{i=0}^{7} x_i[m] * dN_i/dxi_n
                J = jnp.zeros((3, 3))
                for i in range(8):
                    # Outer product: node coordinate (3,) with derivative (3,)
                    J = J + jnp.outer(element_nodes[i], dN_dxi[i])
                detJ = jnp.abs(jnp.linalg.det(J))
                # Map derivatives to physical coordinates:
                # dN_dx: (8,3) where for each node: dN_dx = J^{-1} * dN_dxi (for that node)
                dN_dx = jnp.linalg.solve(J, dN_dxi.T).T  # shape (8,3)
                # Contribution: for conduction the bilinear form is grad(N_i) dot grad(N_j)
                Ke = Ke + k_eff * (dN_dx @ dN_dx.T) * detJ
    return Ke


# -------------------------------
# 4. Mesh generation for a structured grid
# -------------------------------
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


# -------------------------------
# 5. Global assembly of the stiffness matrix and load vector
# -------------------------------
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
        Ke = element_stiffness_matrix(elem_nodes, k_eff)
        # Scatter assembly into the global stiffness matrix:
        for i_local in range(8):
            i_global = elem_nodes_idx[i_local]
            for j_local in range(8):
                j_global = elem_nodes_idx[j_local]
                K_global = K_global.at[i_global, j_global].add(Ke[i_local, j_local])
    return K_global, f_global


# -------------------------------
# 6. Apply boundary conditions
# -------------------------------
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


# -------------------------------
# 7. Solve the system
# -------------------------------
def solve_system(K, f):
    """
    Solve the linear system K*T = f for the nodal temperatures T.
    """
    T = jnp.linalg.solve(K, f)
    return T


# -------------------------------
# 8. Overall FEA pipeline
# -------------------------------
def fea_3d_thermal(nx, ny, nz, lx, ly, lz,
                   base_k, penal, density, h, T_inf,
                   fixed_nodes, fixed_values, convection_nodes, conv_area):
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
      penal      : Penalization exponent for the density.
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
    nodes, elements = generate_mesh(nx, ny, nz, lx, ly, lz)
    K, f = assemble_global_system(nodes, elements, density, base_k, penal)
    K, f = apply_convection_bc(K, f, convection_nodes, h, T_inf, conv_area)
    K, f = apply_dirichlet_bc(K, f, fixed_nodes, fixed_values)
    T = solve_system(K, f)
    return nodes, elements, T


# -------------------------------
# 9. Example usage
# -------------------------------
if __name__ == "__main__":
    # Domain and discretization parameters:
    # nx, ny, nz = 10, 10, 10
    # lx, ly, lz = 1.0, 1.0, 1.0
    nx, ny, nz = 5, 5, 5
    lx, ly, lz = 2.0, 2.0, 2.0
    base_k = 1.0
    penal = 3.0
    # For simplicity, assume all elements are “solid” (density = 1.0)
    nodes_temp, elements_temp = generate_mesh(nx, ny, nz, lx, ly, lz)
    n_elem = elements_temp.shape[0]
    density = jnp.ones(n_elem)

    # Define convection: let the top face (z = lz) have convection.
    tol = 1e-6
    nodes_np = np.array(nodes_temp)
    convection_nodes = np.where(np.abs(nodes_np[:, 2] - lz) < tol)[0]
    # Assume each convection node represents an equal share of the top surface area.
    conv_area = (lx * ly) / ((nx + 1) * (ny + 1))
    h = 10.0
    T_inf = 300.0

    # Define Dirichlet BCs: for example, the bottom face (z=0) is fixed at 300.
    fixed_nodes = np.where(np.abs(nodes_np[:, 2] - 0.0) < tol)[0]
    fixed_values = np.full(fixed_nodes.shape, 300.0)

    # Run the FEA pipeline.
    nodes, elements, T = fea_3d_thermal(
        nx, ny, nz, lx, ly, lz,
        base_k, penal, density, h, T_inf,
        fixed_nodes, fixed_values, convection_nodes, conv_area
    )

    # Convert the resulting temperature field to a NumPy array for further processing or plotting.
    T_np = np.array(T)
    print("Computed nodal temperatures (sample):", T_np[:10])

    # Convert the solution to NumPy arrays for plotting.
    nodes_plot = np.array(nodes)
    T_plot = np.array(T)

    # Plot the FEA results.
    dims = (nx + 1, ny + 1, nz + 1)
    plot_temperature_distribution_pyvista(nodes_plot, T_plot, convection_nodes, fixed_nodes, dims=dims)