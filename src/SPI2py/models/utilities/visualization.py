import numpy as np
import pyvista as pv


def plot_stl_model():
    # TODO: Implement
    pass


def plot_mdbd_model(plotter, positions, radii, color='green'):

    spheres = []
    for i in range(len(positions)):
        sphere = pv.Sphere(center=positions[i], radius=radii[i])
        spheres.append(sphere)

    merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
    plotter.add_mesh(merged, color=color, opacity=0.95)

    plotter.view_xz()
    plotter.background_color = 'white'
    plotter.show_axes()
    # plotter.show_bounds(color='black')
    plotter.show()


def plot_grid(plotter, nx, ny, nz, bounds, spacing):

    # Unpack bounds
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Create an empty uniform grid
    grid = pv.ImageData()

    # Set the grid dimensions
    grid.dimensions = np.array([nx + 1, ny + 1, nz + 1])

    # Set the spacing
    grid.spacing = (spacing, spacing, spacing)

    # Set the origin of the grid to the minimum XYZ coordinates
    grid.origin = (x_min, y_min, z_min)

    plotter.add_mesh(grid, color='lightgrey', show_edges=True, opacity=0.15, lighting=False)


def plot_densities(plotter, densities):

    # # Get the dimensions of the grid
    # m, n, p = densities.shape[0], densities.shape[1], densities.shape[2]
    pass

def plot_bounding_box():
    pass



def plot_problem(prob):
    """
    Plot the model at a given state.
    """

    # Create the plotter
    plotter = pv.Plotter(shape=(1, 2), window_size=(1000, 500))

    # Plot 1: Objects
    plotter.subplot(0, 0)

    # Plot the components
    components = []
    component_colors = []
    for subsystem in prob.model.spatial_config.components._subsystems_myproc:
        positions = prob.get_val(f'spatial_config.components.{subsystem.name}.transformed_sphere_positions')
        radii = prob.get_val(f'spatial_config.components.{subsystem.name}.transformed_sphere_radii')
        color = subsystem.options['color']

        spheres = []
        for position, radius in zip(positions, radii):
            spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))

        merged = pv.MultiBlock(spheres).combine().extract_surface().clean()

        components.append(merged)
        component_colors.append(color)

    for comp, color in zip(components, component_colors):
        plotter.add_mesh(comp, color=color, opacity=0.5)

    # Plot the interconnects
    if 'interconnects' in prob.model.spatial_config._subsystems_allprocs:

        interconnects = []
        interconnect_colors = []
        for subsystem in prob.model.spatial_config.interconnects._subsystems_myproc:

            positions = prob.get_val(f'spatial_config.interconnects.{subsystem.name}.transformed_sphere_positions')
            radii = prob.get_val(f'spatial_config.interconnects.{subsystem.name}.transformed_sphere_radii')
            color = subsystem.options['color']

            # Plot the spheres
            spheres = []
            for position, radius in zip(positions, radii):
                spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))

            # Plot the cylinders
            cylinders = []
            for i in range(len(positions) - 1):
                start = positions[i]
                stop = positions[i + 1]
                radius = radii[i]
                length = np.linalg.norm(stop - start)
                direction = (stop - start) / length
                center = (start + stop) / 2
                cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                cylinders.append(cylinder)

            # merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
            merged_spheres = pv.MultiBlock(spheres).combine().extract_surface().clean()
            merged_cylinders = pv.MultiBlock(cylinders).combine().extract_surface().clean()
            merged = merged_spheres + merged_cylinders

            interconnects.append(merged)
            interconnect_colors.append(color)

        for inter, color in zip(interconnects, interconnect_colors):
            plotter.add_mesh(inter, color=color, lighting=False)

    # Plot 2: The combined density with colored spheres
    plotter.subplot(0, 1)

    # Plot grid
    bounds = prob.model.mesh.options['bounds']
    nx = int(prob.get_val('mesh.n_el_x'))
    ny = int(prob.get_val('mesh.n_el_y'))
    nz = int(prob.get_val('mesh.n_el_z'))
    spacing = float(prob.get_val('mesh.element_length'))
    plot_grid(plotter, nx, ny, nz, bounds, spacing)


    # Plot projections
    pseudo_densities = prob.get_val(f'spatial_config.system.pseudo_densities')
    centers = prob.get_val(f'mesh.centers')

    # Plot the projected pseudo-densities of each element (speed up by skipping near-zero densities)
    density_threshold = 1e-3
    above_threshold_indices = np.argwhere(pseudo_densities > density_threshold)
    for idx in above_threshold_indices:
        n_i, n_j, n_k = idx

        # Calculate the center of the current box
        center = centers[n_i, n_j, n_k]
        density = pseudo_densities[n_i, n_j, n_k]

        if density > 1:
            # Create the box
            box = pv.Cube(center=center, x_length=2*spacing, y_length=2*spacing, z_length=2*spacing)
            plotter.add_mesh(box, color='red', opacity=0.5)
        else:
            # Create the box
            box = pv.Cube(center=center, x_length=spacing, y_length=spacing, z_length=spacing)
            plotter.add_mesh(box, color='black', opacity=density)


    # Configure the plot
    plotter.link_views()
    plotter.view_xy()
    # plotter.view_isometric()
    plotter.show_axes()
    # plotter.show_bounds(color='black')
    # p.background_color = 'white'

    plotter.show()
