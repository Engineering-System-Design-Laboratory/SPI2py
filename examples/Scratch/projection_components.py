import numpy as np
import pyvista as pv
from SPI2py.models.projection.grid import create_grid
from SPI2py.models.mechanics.transformations_rigidbody import transform_points
from SPI2py.models.projection.projection import project_component
from SPI2py.models.utilities.visualization import plot_grid, plot_spheres, plot_AABB, plot_stl_file
from SPI2py.models.projection.grid_kernels import create_uniform_inscription_kernel

# Create grid
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
# S_k = 10.0e-2
S_k = 9.0e-2
xyzr_kernel = np.loadtxt('csvs/mdbd_kernel.csv', delimiter=',')
xyzr_kernel = xyzr_kernel[xyzr_kernel[:, 3] >= S_k]
kernel_pos = xyzr_kernel[:, :3]
kernel_rad = xyzr_kernel[:, 3:4]
# kernel_pos, kernel_rad = create_uniform_inscription_kernel(3)
# kernel_pos = kernel_pos.reshape(-1, 3)
# kernel_rad = kernel_rad.reshape(-1, 1)

# Initialize a primitive kernel
# kernel_pos = np.array([[0, 0, 0]])
# kernel_rad = np.array([[0.5]])  # Inscription radius
# kernel_rad = np.linalg.norm([1, 1, 1])/2  # Circumscription radius


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
densities_be, sample_positions_be, sample_radii_be = project_component(el_centers, el_size, pos_be, rad_be, kernel_pos, kernel_rad)
densities_cdg, sample_positions_cdg, sample_radii_cdg = project_component(el_centers, el_size, pos_cdg, rad_cdg, kernel_pos, kernel_rad)
densities_chp, sample_positions_chp, sample_radii_chp = project_component(el_centers, el_size, pos_chp, rad_chp, kernel_pos, kernel_rad)
# densities_combined = densities_be + densities_cdg + densities_chp
densities_combined = np.minimum(densities_be + densities_cdg + densities_chp, 1.0)  # TODO Implement w/ aggregation


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





