import numpy as np
import pyvista as pv
from create import create_grid
from mdbd import get_aabb_indices
from volume import volume_intersection_two_spheres
from kernel import apply_kernel
from rigid_body_transformation import transform_points
from visualization import plot_grid, plot_spheres, plot_AABB, plot_stl_file


def calculate_pseudo_densities(element_centers, element_size, object_positions, object_radii):
    """
    Projects the points to the mesh and calculates the pseudo-densities

    mesh_positions: (n_el_x, n_el_y, n_el_z, n_mesh_points, 3) tensor
    mesh_radii: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1) tensor

    mesh_positions_expanded: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1, 3) tensor


    object_positions: (n_object_points, 3) tensor
    object_radii: (n_object_points, 1) tensor

    object_positions_expanded: (1, 1, 1, 1, n_object_points, 3) tensor
    object_radii_expanded:     (1, 1, 1, 1, n_object_points) tensor

    pseudo_densities: (n_el_x, n_el_y, n_el_z) tensor
    """

    # Unpack the AABB indices
    i1, i2, j1, j2, k1, k2 = get_aabb_indices(element_centers, element_size,
                                              object_positions, object_radii)

    # Preallocate the pseudo-densities
    n_el_x, n_el_y, n_el_z, _, _ = element_centers.shape
    all_densities = np.zeros((n_el_x, n_el_y, n_el_z))

    # Slice the mesh elements that are within the object's AABB
    element_centers_sliced = element_centers[i1:i2+1, j1:j2+1, k1:k2+1]

    # Apply the kernel to the mesh elements
    # TODO Fix variable scopes
    mesh_positions, mesh_radii = apply_kernel(element_centers_sliced, element_size, kernel_pos, kernel_rad)

    # Calculate the volume of the spheres in each mesh element
    element_sample_volumes = (4 / 3) * np.pi * mesh_radii ** 3
    element_volumes = np.sum(element_sample_volumes, axis=3, keepdims=True)

    # Expand the arrays to allow broadcasting
    object_radii_transposed = object_radii.T
    object_radii_transposed_expanded = np.expand_dims(object_radii_transposed, axis=(0, 1, 2))

    mesh_positions_expanded = np.expand_dims(mesh_positions, axis=(4,))
    object_positions_expanded = np.expand_dims(object_positions, axis=(0, 1, 2, 3))

    # s
    distances = np.linalg.norm(mesh_positions_expanded - object_positions_expanded, axis=-1)

    #
    volume_sample_overlaps = volume_intersection_two_spheres(object_radii_transposed_expanded, mesh_radii, distances)

    #
    volume_element_overlaps = np.sum(volume_sample_overlaps, axis=4, keepdims=True)
    volume_fractions = (volume_element_overlaps / element_volumes).squeeze(4)

    #
    densities = np.sum(volume_fractions, axis=(3,), keepdims=False)

    # Insert the pseudo-densities into the pre-allocated array
    all_densities[i1:i2+1, j1:j2+1, k1:k2+1] = densities

    return all_densities, mesh_positions, mesh_radii


# Define a 5x5x5 grid
# TODO Fix so bounds inclusive (?)
el_size = 0.5
el_centers = create_grid(0, 2, 0, 6.5, 0,  4.5, element_size=el_size)

# Read the mesh kernel
# Slice by minimum radius instead of length to maintain kernel symmetry
# >=10.0e-2 for up to 9 points
# >=9.0e-2 for up to 33 points
# >=6.0e-2 for up to 72 points
# >=4.0e-2 for up to 181 points
# >=3.0e-2 for up to 326 points
# >=2.0e-2 for up to 832 points
# S_k = 4.0e-2
S_k = 10.0e-2
xyzr_kernel = np.loadtxt('csvs/mdbd_kernel.csv', delimiter=',')
xyzr_kernel = xyzr_kernel[xyzr_kernel[:, 3] >= S_k]
kernel_pos = xyzr_kernel[:, :3]
kernel_rad = xyzr_kernel[:, 3:4]

# Initialize a primitive kernel
# kernel_pos = np.array([[0, 0, 0]])
# kernel_rad = np.array([[0.5]])
# kernel_rad = np.array([[0.8660254037844386]])

# Read the part model (Numbers for part 1)
# Slice by minimum radius instead of length to maintain kernel symmetry
# >=4.0e-2 for up to 122 points
# >=3.0e-2 for up to 238 points
# >=2.0e-2 for up to 623 points
# S_c = 4.0e-2
S_c = 4.0e-2

# Part 1
xyzr_be = np.loadtxt('csvs/Bot_Eye_5k_300s.csv', delimiter=',')
xyzr_be = xyzr_be[xyzr_be[:, 3] >= S_c]
pos_be = xyzr_be[:, :3]
rad_be = xyzr_be[:, 3:4]
pos_be_center = np.mean(pos_be, axis=0, keepdims=True)
pos_be = transform_points(pos_be, pos_be_center, translation=(0.625, 0.625, 0.125), rotation=(0, 0, 0))

# Part 2
xyzr_cdg = np.loadtxt('csvs/CogDrivenGear_5k_300s.csv', delimiter=',')
xyzr_cdg = xyzr_cdg[xyzr_cdg[:, 3] >= S_c]
pos_cdg = xyzr_cdg[:, :3]
rad_cdg = xyzr_cdg[:, 3:4]
pos_cdg_center = np.mean(pos_cdg, axis=0, keepdims=True)
pos_cdg = transform_points(pos_cdg, pos_cdg_center, translation=(0.5, 4, 1.5), rotation=(np.pi/2, 0, np.pi/2))

# Part 3
xyzr_chp = np.loadtxt('csvs/CrossHead_Pin_5k_300s.csv', delimiter=',')
xyzr_chp = xyzr_chp[xyzr_chp[:, 3] >= S_c]
pos_chp = xyzr_chp[:, :3]
rad_chp = xyzr_chp[:, 3:4]
pos_chp_center = np.mean(pos_chp, axis=0, keepdims=True)
pos_chp = transform_points(pos_chp, pos_chp_center, translation=(0, 0.5, 1.75), rotation=(0, 0, 0))

# Calculate the pseudo-densities
densities_be, sample_positions_be, sample_radii_be = calculate_pseudo_densities(el_centers, el_size, pos_be, rad_be)
densities_cdg, sample_positions_cdg, sample_radii_cdg = calculate_pseudo_densities(el_centers, el_size, pos_cdg, rad_cdg)
densities_chp, sample_positions_chp, sample_radii_chp = calculate_pseudo_densities(el_centers, el_size, pos_chp, rad_chp)
densities_combined = densities_be + densities_cdg + densities_chp

# Plot
plotter = pv.Plotter(shape=(2, 3), window_size=(1500, 500))

# Plot the grid without the kernel
plot_grid(plotter, (0, 0), el_centers, el_size, densities=None)
plot_stl_file(plotter, (0, 0), 'models/Bot_Eye_scaled.stl', translation=(0.625, 0.625, 0.125), rotation=(0, 0, 0))
plot_stl_file(plotter, (0, 0), 'models/CogDrivenGear_scaled.stl', translation=(0.5, 4, 1.5), rotation=(90, 0, 90))
plot_stl_file(plotter, (0, 0), 'models/CrossHead_Pin_scaled.stl', translation=(0, 0.5, 1.75))
plot_AABB(plotter, (0, 0), pos_be, rad_be, color='blue')
plot_AABB(plotter, (0, 0), pos_cdg, rad_cdg, color='orange')
plot_AABB(plotter, (0, 0), pos_chp, rad_chp, color='red')

# Plot the grid with the kernel
plot_AABB(plotter, (1, 1), sample_positions_be, sample_radii_be, color='black', opacity=0.0)
plot_AABB(plotter, (1, 1), sample_positions_cdg, sample_radii_cdg, color='black', opacity=0.0)
plot_AABB(plotter, (1, 1), sample_positions_chp, sample_radii_chp, color='black', opacity=0.0)
plot_spheres(plotter, (1, 1), sample_positions_be, sample_radii_be, 'blue', opacity=0.5)
plot_spheres(plotter, (1, 1), sample_positions_cdg, sample_radii_cdg, 'orange', opacity=0.5)
plot_spheres(plotter, (1, 1), sample_positions_chp, sample_radii_chp, 'red', opacity=0.5)
plot_spheres(plotter, (0, 1), pos_be, rad_be, 'blue', opacity=0.5)
plot_spheres(plotter, (0, 1), pos_cdg, rad_cdg, 'orange', opacity=0.5)
plot_spheres(plotter, (0, 1), pos_chp, rad_chp, 'red', opacity=0.5)
plot_stl_file(plotter, (0, 1), 'models/Bot_Eye_scaled.stl', translation=(0.625, 0.625, 0.125), rotation=(0, 0, 0), opacity=0.5)
plot_stl_file(plotter, (0, 1), 'models/CogDrivenGear_scaled.stl', translation=(0.5, 4, 1.5), rotation=(90, 0, 90), opacity=0.5)
plot_stl_file(plotter, (0, 1), 'models/CrossHead_Pin_scaled.stl', translation=(0, 0.5, 1.75), opacity=0.5)

# Plot the grid without the kernel
plot_grid(plotter, (0, 2), el_centers, el_size, densities=densities_combined)

plotter.show_axes()
plotter.link_views()
plotter.show()

# # # TODO Add rotation
# # # Now measure fluctuations
# pos_start = np.array([0.5, 0.5, 0])
# pos_stop = np.array([0.75, 0.75, 0.25])
# n_steps = 5
#
# def measure_fluctuations(sphere_pos, sphere_rad, pos_start, pos_stop, n_steps):
#     """
#     Measure the fluctuations in the pseudo-densities
#     """
#     # Calculate the step size
#     step_size = np.array((pos_stop - pos_start) / n_steps)
#
#     # Preallocate the fluctuations
#     # fluctuations = np.zeros(n_steps)
#     masses = np.zeros(n_steps)
#     positions = [sphere_pos]
#     sample_positions_all = []
#     sample_radii_all = []
#     for i in range(n_steps):
#         # Calculate the new positions
#         sphere_pos += step_size
#
#         # Calculate the pseudo-densities
#         densities, sample_positions, sample_radii = calculate_pseudo_densities(el_centers, el_size, sphere_pos, sphere_rad)
#
#         positions.append(sphere_pos.copy())
#         sample_positions_all.append(sample_positions)
#         sample_radii_all.append(sample_radii)
#
#         mass = np.sum(densities)
#         masses[i] = mass
#
#     return masses, positions, sample_positions_all, sample_radii_all
#
# masses, positions, sample_positions_all, sample_radii_all = measure_fluctuations(pos, rad, pos_start, pos_stop, n_steps)
#
# # fluctuation = np.std(masses)
# max_fluctuation = np.max(masses) - np.min(masses)
# avg_mass = np.mean(masses)
# print(f"Max fluctuation: {max_fluctuation/avg_mass*100:.2f}%")

# print(f"Fluctuation: {fluctuation}")

# # Plot
# plotter = pv.Plotter(shape=(1, n_steps), window_size=(1500, 500))
#
# for i, (pos, samp_pos, samp_rad) in enumerate(zip(positions, sample_positions_all, sample_radii_all)):
#     plot_grid(plotter, (0, i), el_centers, el_size, densities=None)
#     plot_grid_spheres(plotter, (0, i), samp_pos, samp_rad)
#     plot_spheres(plotter, (0, i), pos, rad)
#     plotter.add_text(f"Step {i}", position='upper_edge', font_size=14)
#     plotter.show_bounds(all_edges=True)
#
#
# plotter.link_views()
# plotter.show()





