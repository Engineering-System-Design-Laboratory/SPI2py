import numpy as np
import pyvista as pv
import matplotlib.cm as mpl_cm
from ..geometry.cylinders import create_cylinders
from ..geometry.spheres import get_aabb_bounds


def plot_grid(plotter,
              subplot_index,
              centers, size,
              densities=None,
              min_opacity=5e-3):

    # Create the subplot
    plotter.subplot(*subplot_index)
    plotter.render_window.SetMultiSamples(0)

    # Plot the bounding box (unchanged)
    flat_centers = centers.reshape(-1, 3)
    x_min, y_min, z_min = np.min(flat_centers - size / 2, axis=0)
    x_max, y_max, z_max = np.max(flat_centers + size / 2, axis=0)
    aabb = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))
    plotter.add_mesh(aabb, color='black', style='wireframe')

    # Create a PolyData point cloud from the centers
    points = pv.PolyData(flat_centers)

    # Create a cube template: a unit cube centered at (0,0,0)
    cube_template = pv.Cube(center=(0, 0, 0), x_length=1, y_length=1, z_length=1)
    # Scale the cube so that its edge lengths equal the desired size.
    cube_template.points *= size

    # Use the glyph filter to replicate the cube at each point.
    # Since each cube is already scaled to 'size', we disable further scaling.
    glyphs = points.glyph(orient=False, scale=False, geom=cube_template)

    # If densities are provided, create an RGBA array for per-glyph opacity.
    if densities is not None:
        flat_densities = densities.flatten()
        # Clamp densities to the range [min_opacity, 1.0]
        clamped = np.maximum(min_opacity, np.minimum(flat_densities, 1.0))
        # Convert clamped densities to alpha values in 0-255.
        alphas = (clamped * 255).astype(np.uint8)
        # Determine the number of points per glyph.
        n_input = flat_centers.shape[0]
        npts_per_glyph = glyphs.n_points // n_input
        # Repeat each alpha value so that each glyph's points get the same alpha.
        new_alphas = np.repeat(alphas, npts_per_glyph)

        # Create an RGBA array for all points in the glyph mesh.
        rgba = np.zeros((glyphs.n_points, 4), dtype=np.uint8)
        rgba[:, 3] = new_alphas  # Set the alpha channel.
        # R, G, B remain zero for black.
        glyphs.point_data["RGBA"] = rgba  # Use point_data here

        # Add the glyph mesh using the RGBA values.
        plotter.add_mesh(glyphs, rgba=True, lighting=False)
    else:
        plotter.add_mesh(glyphs, color='black', opacity=min_opacity, lighting=False)


def plot_spheres(plotter, subplot_index, positions, radii, color, opacity=0.15):

    # Create the subplot
    plotter.subplot(*subplot_index)
    plotter.render_window.SetMultiSamples(0)

    # Create a sphere template.
    sphere_template = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8)

    # Flatten positions and radii
    flat_positions = positions.reshape(-1, 3)
    flat_radii = radii.flatten()

    # Create a point cloud from positions.
    points = pv.PolyData(np.array(flat_positions))
    # Add radii as a scalar field that the glyph filter will use for scaling.
    points["scale"] = flat_radii

    # Generate glyphs:
    # The 'scale' argument here tells the glyph filter to use the "scale" array for scaling.
    glyphs = points.glyph(orient=False, scale="scale", geom=sphere_template, factor=1.0)

    # Add the combined glyph mesh to the plotter
    plotter.add_mesh(glyphs, color=color, opacity=opacity)


def plot_capsules(plotter, subplot_index, cyl_control_points, cyl_radius, color, opacity=0.875):
    cyl_starts, cyl_stops, cyl_radii = create_cylinders(cyl_control_points, cyl_radius)

    # Create the subplot
    plotter.subplot(*subplot_index)
    plotter.render_window.SetMultiSamples(0)

    for cyl_start, cyl_stop, cyl_radius in zip(cyl_starts, cyl_stops, cyl_radii):
        # Plot the spheres
        plot_spheres(plotter, subplot_index, cyl_start, cyl_radius, color, opacity)
        plot_spheres(plotter, subplot_index, cyl_stop, cyl_radius, color, opacity)

        # Plot the cylinders
        length = np.linalg.norm(cyl_stop - cyl_start)
        direction = (cyl_stop - cyl_start) / length
        center = (cyl_start + cyl_stop) / 2
        cylinder = pv.Cylinder(center=center, direction=direction, radius=cyl_radius, height=length)
        plotter.add_mesh(cylinder, color=color, opacity=opacity, lighting=False)

    plotter.add_text("Pipe Segments", position='upper_edge', font_size=14)
    plotter.show_bounds(all_edges=True)


def plot_AABB(plotter, subplot_index, centers, radii, color, opacity=0.25):
    # Create the subplot
    plotter.subplot(*subplot_index)
    plotter.render_window.SetMultiSamples(0)

    # Get the AABB bounds
    x_min, x_max, y_min, y_max, z_min, z_max = get_aabb_bounds(centers, radii)

    # Create the AABB
    aabb = pv.Box([x_min, x_max, y_min, y_max, z_min, z_max])

    # Add the AABB to the plot
    plotter.add_mesh(aabb, color='black', style='wireframe', lighting=False)

    # Add the AABB to the plot
    plotter.add_mesh(aabb, color=color, opacity=opacity, lighting=False)


def plot_stl_file(plotter, subplot_index, stl_file_path, translation=(0, 0, 0), rotation=(0, 0, 0), scaling=1,
                  opacity=0.5):
    """
    Plots an STL file with an optional translation.

    Parameters:
    - plotter: The PyVista plotter instance.
    - subplot_index: Tuple specifying the subplot location.
    - stl_file_path: Path to the STL file.
    - translation: Tuple (x, y, z) to shift the STL mesh.
    """
    # Create the subplot
    plotter.subplot(*subplot_index)
    plotter.render_window.SetMultiSamples(0)

    # Load the STL file
    mesh = pv.read(stl_file_path)

    # Apply scaling if needed
    if scaling != 1:
        mesh.scale(scaling, inplace=True)

    # Apply translation if needed
    if translation != (0, 0, 0):
        mesh.translate(translation, inplace=True)

    # Apply rotation if needed
    if rotation != (0, 0, 0):
        rx, ry, rz = rotation
        # Get the center of the mesh
        center = mesh.center
        # Rotate around the mesh's center
        if rx != 0:
            mesh.rotate_x(rx, point=center, inplace=True)
        if ry != 0:
            mesh.rotate_y(ry, point=center, inplace=True)
        if rz != 0:
            mesh.rotate_z(rz, point=center, inplace=True)

    # Add the mesh to the plotter
    plotter.add_mesh(mesh, color='lightgray', opacity=opacity, lighting=False)


# def element_to_node_density(element_density, dims):
#     """
#     Convert an element-centered density field (shape (nx, ny, nz))
#     into a nodal density field (shape dims, where dims = (nx+1, ny+1, nz+1))
#     by averaging the densities of adjacent elements.
#     """
#     nx, ny, nz = element_density.shape
#     # dims must equal (nx+1, ny+1, nz+1)
#     if dims != (nx + 1, ny + 1, nz + 1):
#         raise ValueError("dims must be (density.shape[0]+1, density.shape[1]+1, density.shape[2]+1)")
#
#     node_density = np.zeros(dims)
#     count = np.zeros(dims)
#     for i in range(nx):
#         for j in range(ny):
#             for k in range(nz):
#                 # The element at (i,j,k) contributes to 8 nodes:
#                 node_density[i:i + 2, j:j + 2, k:k + 2] += element_density[i, j, k]
#                 count[i:i + 2, j:j + 2, k:k + 2] += 1
#     count[count == 0] = 1
#     return node_density / count
#
#
# def plot_temperature_distribution(plotter,
#                                   subplot_index,
#                                   nodes,
#                                   T,
#                                   density,
#                                   convection_nodes,
#                                   fixed_nodes,
#                                   dims,
#                                   cmap="inferno",
#                                   min_opacity=5e-3):
#     """
#     Visualize the 3D temperature distribution on a structured grid using PyVista,
#     and modulate the opacity according to a provided density field.
#
#     Parameters:
#       plotter          : PyVista plotter instance.
#       subplot_index    : Tuple specifying the subplot location.
#       nodes            : NumPy array of shape (n_nodes, 3) with node coordinates.
#       T                : NumPy array of nodal temperatures (n_nodes,).
#       density          : NumPy array of element densities, shape (nx, ny, nz).
#       convection_nodes : 1D NumPy array of indices for nodes on the convection boundary.
#       fixed_nodes      : 1D NumPy array of indices for nodes with fixed (Dirichlet) conditions.
#       dims             : Tuple (nx+1, ny+1, nz+1) that exactly describes the nodal grid.
#       cmap             : Name of the colormap to use (default "inferno").
#       min_opacity      : Minimum opacity to assign to very low density.
#     """
#     # Check that the grid dimensions match the nodes.
#     expected_nodes = np.prod(dims)
#     if nodes.shape[0] != expected_nodes:
#         raise ValueError(f"Number of nodes ({nodes.shape[0]}) does not match product of dims ({expected_nodes}).")
#     if T.size != expected_nodes:
#         raise ValueError(
#             f"Temperature array size ({T.size}) does not match expected number of nodes ({expected_nodes}).")
#
#     # Reshape nodes into structured grid.
#     try:
#         grid_points = nodes.reshape(dims + (3,))
#     except ValueError:
#         raise ValueError("Nodes cannot be reshaped to the provided dimensions.")
#
#     grid = pv.StructuredGrid()
#     grid.points = grid_points.reshape(-1, 3)
#     grid.dimensions = dims
#     grid["Temperature"] = T
#
#     # Convert the element-centered density to a nodal density field.
#     nodal_density = element_to_node_density(density, dims)
#     flat_nodal_density = nodal_density.flatten()
#     if flat_nodal_density.shape[0] != expected_nodes:
#         raise ValueError(
#             f"Converted nodal density has length {flat_nodal_density.shape[0]}, expected {expected_nodes}.")
#
#     # Normalize the temperature field for colormap mapping.
#     T_min, T_max = np.min(T), np.max(T)
#     T_norm = (T - T_min) / (T_max - T_min + 1e-8)
#     colormap = mpl_cm.get_cmap(cmap)
#     # Map normalized temperature to RGBA colors. This should return an array of shape (expected_nodes, 4).
#     colors = colormap(T_norm)  # values in [0,1]
#
#     # Override the alpha channel with the nodal density.
#     # Clamp nodal density values to [min_opacity, 1.0].
#     alphas = np.clip(flat_nodal_density, min_opacity, 1.0)
#     # Check if colors has 4 channels.
#     if colors.shape[1] != 4:
#         # If the colormap returns only 3 channels, add an alpha channel.
#         colors = np.concatenate([colors, np.ones((colors.shape[0], 1))], axis=1)
#     colors[:, 3] = alphas  # Set the alpha channel
#
#     # Convert to 8-bit integer format.
#     rgba = (colors * 255).astype(np.uint8)
#     grid.point_data["RGBA"] = rgba
#
#     # Add the grid mesh using the RGBA values.
#     plotter.add_mesh(grid, rgba=True, show_edges=True)
#
#     # Overlay markers for convection and fixed nodes.
#     conv_points = nodes[convection_nodes]
#     fixed_points = nodes[fixed_nodes]
#     conv_poly = pv.PolyData(conv_points)
#     fixed_poly = pv.PolyData(fixed_points)
#     plotter.add_mesh(conv_poly, color="blue", point_size=10, render_points_as_spheres=True, label="Convection BC")
#     plotter.add_mesh(fixed_poly, color="red", point_size=10, render_points_as_spheres=True, label="Dirichlet BC")
#     plotter.add_legend()







def plot_temperature_distribution(plotter,
                                  subplot_index,
                                  nodes,
                                  T,
                                  convection_nodes,
                                  fixed_nodes,
                                  dims=None,
                                  cmap="rainbow",
                                  opacity=0.25,
                                  climits=(0, 300)):
    """
    Visualize the 3D temperature distribution on a structured grid using PyVista.

    Parameters:
        plotter          : PyVista plotter instance.
        subplot_index    : Tuple (i, j) specifying the subplot location.
      nodes            : NumPy array of shape (n_nodes,3) with node coordinates.
      T                : NumPy array of nodal temperatures.
      convection_nodes : 1D NumPy array of indices for nodes on the convection boundary.
      fixed_nodes      : 1D NumPy array of indices for nodes with Dirichlet conditions.
      dims             : Tuple (nx+1, ny+1, nz+1) defining grid dimensions. If None, it is inferred.
      origin           : Grid origin (default (0,0,0)).
      cmap             : Colormap for temperature (default "inferno").
      opacity          : Mesh opacity.
    """

    # Create the subplot
    plotter.subplot(*subplot_index)
    plotter.render_window.SetMultiSamples(0)

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

    # # Add dummy scalar values to enforce the colormap range
    # T_with_limits = np.concatenate(([climits[0]], T, [climits[1]]))
    # grid["Temperature"] = T_with_limits

    # Setup PyVista plotter.
    vol = plotter.add_volume(grid, scalars="Temperature", cmap=cmap, clim=(0,1), opacity=opacity, show_scalar_bar=True, scalar_bar_args={'title': 'Temperature'})
    plotter.add_mesh(conv_poly, color="blue", point_size=10, render_points_as_spheres=True, label="Convection BC")
    plotter.add_mesh(fixed_poly, color="red", point_size=10, render_points_as_spheres=True, label="Dirichlet BC")
    plotter.add_legend()

    # Force the scalar range on the volume mapper
    vol.mapper.scalar_range = climits

    vol.prop.interpolation_type = 'linear'

# def plot_problem(prob):
#     """
#     Plot the model at a given state.
#     """
#
#     # Create the plotter
#     plotter = pv.Plotter(shape=(1, 2), window_size=(1000, 500))
#
#     # Plot 1: Objects
#     plotter.subplot(0, 0)
#
#     # Plot the components
#     components = []
#     component_colors = []
#     for subsystem in prob.model.spatial_config.components._subsystems_myproc:
#         positions = prob.get_val(f'spatial_config.components.{subsystem.name}.transformed_sphere_positions')
#         radii = prob.get_val(f'spatial_config.components.{subsystem.name}.transformed_sphere_radii')
#         color = subsystem.options['color']
#
#         spheres = []
#         for position, radius in zip(positions, radii):
#             spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))
#
#         merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
#
#         components.append(merged)
#         component_colors.append(color)
#
#     for comp, color in zip(components, component_colors):
#         plotter.add_mesh(comp, color=color, opacity=0.5)
#
#     # Plot the interconnects
#     if 'interconnects' in prob.model.spatial_config._subsystems_allprocs:
#
#         interconnects = []
#         interconnect_colors = []
#         for subsystem in prob.model.spatial_config.interconnects._subsystems_myproc:
#
#             positions = prob.get_val(f'spatial_config.interconnects.{subsystem.name}.transformed_sphere_positions')
#             radii = prob.get_val(f'spatial_config.interconnects.{subsystem.name}.transformed_sphere_radii')
#             color = subsystem.options['color']
#
#             # Plot the spheres
#             spheres = []
#             for position, radius in zip(positions, radii):
#                 spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))
#
#             # Plot the cylinders
#             cylinders = []
#             for i in range(len(positions) - 1):
#                 start = positions[i]
#                 stop = positions[i + 1]
#                 radius = radii[i]
#                 length = np.linalg.norm(stop - start)
#                 direction = (stop - start) / length
#                 center = (start + stop) / 2
#                 cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
#                 cylinders.append(cylinder)
#
#             # merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
#             merged_spheres = pv.MultiBlock(spheres).combine().extract_surface().clean()
#             merged_cylinders = pv.MultiBlock(cylinders).combine().extract_surface().clean()
#             merged = merged_spheres + merged_cylinders
#
#             interconnects.append(merged)
#             interconnect_colors.append(color)
#
#         for inter, color in zip(interconnects, interconnect_colors):
#             plotter.add_mesh(inter, color=color, lighting=False)
#
#     # Plot 2: The combined density with colored spheres
#     plotter.subplot(0, 1)
#
#     # Plot grid
#     bounds = prob.model.mesh.options['bounds']
#     nx = int(prob.get_val('mesh.n_el_x'))
#     ny = int(prob.get_val('mesh.n_el_y'))
#     nz = int(prob.get_val('mesh.n_el_z'))
#     spacing = float(prob.get_val('mesh.element_length'))
#     plot_grid(plotter, nx, ny, nz, bounds, spacing)
#
#
#     # Plot projections
#     pseudo_densities = prob.get_val(f'spatial_config.system.pseudo_densities')
#     centers = prob.get_val(f'mesh.centers')
#
#     # Plot the projected pseudo-densities of each element (speed up by skipping near-zero densities)
#     density_threshold = 1e-3
#     above_threshold_indices = np.argwhere(pseudo_densities > density_threshold)
#     for idx in above_threshold_indices:
#         n_i, n_j, n_k = idx
#
#         # Calculate the center of the current box
#         center = centers[n_i, n_j, n_k]
#         density = pseudo_densities[n_i, n_j, n_k]
#
#         if density > 1:
#             # Create the box
#             box = pv.Cube(center=center, x_length=2*spacing, y_length=2*spacing, z_length=2*spacing)
#             plotter.add_mesh(box, color='red', opacity=0.5)
#         else:
#             # Create the box
#             box = pv.Cube(center=center, x_length=spacing, y_length=spacing, z_length=spacing)
#             plotter.add_mesh(box, color='black', opacity=density)
#
#
#     # Configure the plot
#     plotter.link_views()
#     plotter.view_xy()
#     # plotter.view_isometric()
#     plotter.show_axes()
#     # plotter.show_bounds(color='black')
#     # p.background_color = 'white'
#
#     plotter.show()
