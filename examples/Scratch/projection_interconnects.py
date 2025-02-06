import numpy as np
import pyvista as pv

from SPI2py.models.projection.grid import create_grid
from SPI2py.models.projection.grid_kernels import create_uniform_kernel
from SPI2py.models.projection.projection import project_interconnect
from SPI2py.models.utilities.visualization import plot_grid, plot_spheres, plot_capsules, plot_AABB


# Create grid
el_size = 0.5
# el_size = 0.25
n, m, o = 3, 6, 3
el_centers = create_grid(0,4,0,6,0,2, element_size=el_size)

# Read the mesh kernel
# Slice by minimum radius instead of length to maintain kernel symmetry
# S_k = 9.0e-2  # >=9.0e-2 for up to 33 points
# S_k = 6.0e-2  # >=6.0e-2 for up to 73 points
# xyzr_kernel = np.loadtxt('csvs/mdbd_kernel.csv', delimiter=',')
# # xyzr_kernel = xyzr_kernel[0:1]
# xyzr_kernel = xyzr_kernel[0:6]
# # xyzr_kernel = xyzr_kernel[xyzr_kernel[:, 3] >= S_k]
# kernel_pos = xyzr_kernel[:, :3]
# kernel_rad = xyzr_kernel[:, 3:4]
kernel_pos, kernel_rad = create_uniform_kernel(3, mode='circumscription')
# kernel_pos, kernel_rad = create_uniform_kernel(3, mode='inscription')
kernel_pos = kernel_pos.reshape(-1, 3)
kernel_rad = kernel_rad.reshape(-1, 1)

# Create line segment arrays
# cyl_control_points = np.array([[0.25, 0.25, 2], [2, 4, 1]])
# cyl_control_points = np.array([[0.25, 0.25, 2], [2, 2, 2]])
# cyl_radius = np.array([0.25])
cyl_control_points = np.array([[0.25, 0.25, 2], [2, 2, 1], [2, 4, 1]])
cyl_radius = np.array([0.25])
# cyl_control_points = np.array([[0, 0, 0], [2, 0, 0], [2, 4, 0]])
# cyl_radius = np.array([0.25])


# Calculate the densities
densities, sample_positions, sample_radii = project_interconnect(el_centers, el_size, cyl_control_points, cyl_radius, kernel_pos, kernel_rad)

# # Plot
# plotter = pv.Plotter(shape=(2, 2), window_size=(1500, 500))
#
# # Plot the interconnect
# plot_grid(plotter, (0, 0), el_centers, el_size, densities=None)
# plot_capsules(plotter, (0, 0), cyl_control_points, cyl_radius, 'blue')
# plot_AABB(plotter, (0, 0), cyl_control_points, cyl_radius, color='blue')
#
# # Plot the kernel
# plot_grid(plotter, (1, 0), el_centers, el_size, densities=None)
# plot_spheres(plotter, (1, 0), sample_positions, sample_radii, 'lightgray')
#
# # Plot the pseudo-densities
# plot_grid(plotter, (1, 1), el_centers, el_size, densities=densities)
#
# # # Plot the grid with the kernel
# # plot_grid(plotter, (0, 1), el_centers, el_size, densities=None)
# # plot_spheres(plotter, (0, 1), sample_positions, sample_radii, 'lightgray')
# # plot_capsules(plotter, (0, 1), cyl_control_points, cyl_radius, 'blue')
#
# plotter.show_bounds(all_edges=True, location='outer')
# plotter.show_axes()
# plotter.link_views()
# plotter.show()

