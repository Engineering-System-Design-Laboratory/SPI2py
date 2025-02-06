import numpy as np
import pyvista as pv
from ..geometry.cylinders import create_cylinders
from ..geometry.spheres import get_aabb_bounds


def plot_grid(plotter, subplot_index, centers, size, densities=None, min_opacity=5e-3):
    plotter.subplot(*subplot_index)
    plotter.render_window.SetMultiSamples(0)

    # Plot the bounding box (unchanged)
    flat_centers = centers.reshape(-1, 3)
    x_min, y_min, z_min = np.min(flat_centers - size/2, axis=0)
    x_max, y_max, z_max = np.max(flat_centers + size/2, axis=0)
    aabb = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))
    plotter.add_mesh(aabb, color='black', style='wireframe')

    # Create a PolyData point cloud from the centers
    points = pv.PolyData(flat_centers)

    # Create a cube template: a unit cube centered at (0,0,0)
    cube_template = pv.Cube(center=(0, 0, 0), x_length=1, y_length=1, z_length=1)
    # Scale the cube so that its edge lengths equal the desired size.
    cube_template.points *= size

    # If densities are provided, assign them as a scalar array.
    # Note: glyphs don’t directly support per-glyph opacity.
    if densities is not None:
        flat_densities = densities.flatten()
        # Clamp densities to the range [min_opacity, 1.0]
        opacities = np.maximum(min_opacity, np.minimum(flat_densities, 1.0))
        points["opacity"] = opacities

    # Use the glyph filter to replicate the cube at each point.
    # Since each cube is already scaled to 'size', we disable further scaling.
    glyphs = points.glyph(orient=False, scale=False, geom=cube_template)

    # Add the combined glyph mesh.
    plotter.add_mesh(glyphs, color='black', opacity=min_opacity, lighting=False)


def plot_spheres(plotter, subplot_index, positions, radii, color, opacity=0.15):
    # Create the subplot
    plotter.subplot(*subplot_index)
    plotter.render_window.SetMultiSamples(0)

    # Create a sphere template.
    sphere_template = pv.Sphere(theta_resolution=8, phi_resolution=8)

    # Flatten positions and radii
    flat_positions = positions.reshape(-1, 3)
    flat_radii = radii.flatten()

    # Create a point cloud from positions.
    points = pv.PolyData(flat_positions)
    # Add radii as a scalar field that the glyph filter will use for scaling.
    points["scale"] = flat_radii

    # Generate glyphs:
    # The 'scale' argument here tells the glyph filter to use the "scale" array for scaling.
    glyphs = points.glyph(orient=False, scale="scale", geom=sphere_template)

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


def plot_stl_file(plotter, subplot_index, stl_file_path, translation=(0, 0, 0), rotation=(0, 0, 0), scaling=1, opacity=0.8):
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
