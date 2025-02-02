import numpy as np
import pyvista as pv
from mesh_kernels import mdbd_9_kernel_positions as kernel_pos
from mesh_kernels import mdbd_9_kernel_radii as kernel_rad
from create import create_grid
from visualization import plot_grid, plot_grid_spheres

def apply_kernel(element_centers, element_sizes, kernel_positions, kernel_radii):

    # Expand the arrays to allow broadcasting
    element_centers_expanded = np.expand_dims(element_centers, axis=(3,))
    element_size_expanded = np.expand_dims(element_sizes, axis=(3, 4))
    kernel_pos_expanded = np.expand_dims(kernel_positions, axis=(0, 1, 2))
    kernel_rad_expanded = np.expand_dims(kernel_radii, axis=(0, 1, 2, 4))

    # Broadcast the kernel positions and radii to the grid
    all_positions = element_centers_expanded + (element_size_expanded * kernel_pos_expanded)
    all_radii = element_size_expanded * kernel_rad_expanded

    return all_positions, all_radii



# Define a 2x2x2 grid
el_length = 1.0
el_centers, el_sizes = create_grid(2, 2, 2, spacing=el_length)

# Convert the kernel positions and radii to numpy arrays
kernel_pos = np.array(kernel_pos)
kernel_rad = np.array(kernel_rad)

# Define the AABB
x_indices = [0, 1]
y_indices = [0, 1]
z_indices = [0, 1]
xx, yy, zz = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
aabb_indices = (xx, yy, zz)

# Index
el_centers = el_centers[aabb_indices]
el_sizes = el_sizes[aabb_indices]

# Apply the kernel to the grid
all_pos, all_rad = apply_kernel(el_centers, el_sizes, kernel_pos, kernel_rad)

# Plot
plotter = pv.Plotter(shape=(2, 2), window_size=(1500, 500))

# Plot the grid without the kernel
plot_grid(plotter, (0, 0), el_centers[0, 0, 0].reshape(1, 1, 1, 3), el_length)

# Plot the grid with the kernel
plot_grid_spheres(plotter, (0, 1), all_pos[0, 0, 0].reshape(1, 1, 1, -1, 3), all_rad[0, 0, 0].reshape(1, 1, 1, -1, 1))


# Plot the grid without the kernel
plot_grid(plotter, (1, 0), el_centers, el_length)

# Plot the grid with the kernel
plot_grid_spheres(plotter, (1, 1), all_pos, all_rad)

plotter.link_views()
plotter.show()

# # Save the plot
# plotter.save_graphic('figures/grid_with_kernel.png')


# # TODO Transpose later
# all_rad = np.transpose(all_rad, axes=(0, 1, 2, 4, 3))












