import numpy as np
import pyvista as pv
from create import create_grid, create_cylinders
from kernel import apply_kernel
from mdbd import get_aabb_indices
from scripts.distance import signed_distance
from visualization import plot_grid, plot_spheres, plot_AABB, plot_capsules


def regularized_Heaviside(x):
    H_tilde = 0.5 + 0.75 * x - 0.25 * x ** 3  # EQ 3 in 3D
    return H_tilde


def density(phi_b, r):
    ratio = phi_b / r
    rho = np.where(ratio < -1, 0,
                   np.where(ratio > 1, 1,
                            regularized_Heaviside(ratio)))
    return rho


def calculate_densities(el_centers, el_size,
                        cyl_control_points, cyl_radius):
    """
    Projects the points to the mesh and calculates the pseudo-densities

    mesh_positions: (n_el_x, n_el_y, n_el_z, n_mesh_points, 3) tensor
    mesh_radii: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1) tensor

    mesh_positions_expanded: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1, 3) tensor

    cylinder_starts: (n_segments, 3) tensor
    cylinder_stops: (n_segments, 3) tensor
    cylinder_radii: (n_segments, 1) tensor

    # TODO Fix for Q cylinders
    cylinder_starts_expanded: (1, 1, 1, n_segments, 3) tensor
    cylinder_stops_expanded: (1, 1, 1, n_segments, 3) tensor
    cylinder_radii_expanded: (1, 1, 1, n_segments, 1) tensor

    pseudo_densities: (n_el_x, n_el_y, n_el_z) tensor
    """

    # Create the cylinders
    cyl_starts, cyl_stops, cyl_rad = create_cylinders(cyl_control_points, cyl_radius)

    # Unpack the AABB indices
    # i1, i2, j1, j2, k1, k2 = get_aabb_indices(el_centers, el_size, cyl_control_points, cyl_radius)


    # Preallocate the pseudo-densities
    n_el_x, n_el_y, n_el_z, _, _ = el_centers.shape
    # all_densities = np.zeros((n_el_x, n_el_y, n_el_z))

    # Slice the mesh elements that are within the object's AABB

    # Apply the kernel to the mesh elements
    # TODO Fix variable scopes
    mesh_positions, mesh_radii = apply_kernel(el_centers, el_size, kernel_pos, kernel_rad)

    # Calculate the volume of the spheres in each mesh element
    element_sample_volumes = (4 / 3) * np.pi * mesh_radii ** 3
    element_volumes = np.sum(element_sample_volumes, axis=3, keepdims=True)


    # Expand the arrays to allow broadcasting
    cyl_rad_transposed = cyl_rad.T
    cyl_rad_transposed_expanded = np.expand_dims(cyl_rad_transposed, axis=(0, 1, 2))

    mesh_positions_expanded = np.expand_dims(mesh_positions, axis=(4,))
    cyl_starts_expanded = np.expand_dims(cyl_starts, axis=(0, 1, 2))
    cyl_stops_expanded = np.expand_dims(cyl_stops, axis=(0, 1, 2))

    # Vectorized signed distance and density calculations using your distance function
    phi = signed_distance(mesh_positions_expanded, cyl_starts_expanded, cyl_stops_expanded, cyl_rad_transposed_expanded)

    # Fix rho for mesh_radii?
    rho = density(phi, mesh_radii)

    # Sum densities across all cylinders

    # Combine the pseudo densities for all cylinders in each kernel sphere
    # Collapse the last axis to get the combined density for each kernel sphere
    combined_density = np.sum(rho, axis=4, keepdims=False)

    # Combine the pseudo densities for all kernel spheres in one grid
    combined_density = np.sum(combined_density, axis=3, keepdims=False)

    # Clip
    # combined_density = np.clip(combined_density, 0, 1)

    return combined_density, mesh_positions, mesh_radii


# Create grid
el_size = 0.5
n, m, o = 3, 5, 1
el_centers = create_grid(0,3,0,5,0,1, element_size=el_size)

# TODO: Appears to work for single kernel, but not for multiple kernels
# Read the mesh kernel
# Slice by minimum radius instead of length to maintain kernel symmetry
# S_k = 9.0e-2  # >=9.0e-2 for up to 33 points
# S_k = 6.0e-2  # >=6.0e-2 for up to 73 points
xyzr_kernel = np.loadtxt('csvs/mdbd_kernel.csv', delimiter=',')
xyzr_kernel = xyzr_kernel[0:1]
# xyzr_kernel = xyzr_kernel[0:6]
# xyzr_kernel = xyzr_kernel[xyzr_kernel[:, 3] >= S_k]
kernel_pos = xyzr_kernel[:, :3]
kernel_rad = xyzr_kernel[:, 3:4]

# Create line segment arrays
cyl_control_points = np.array([[0, 0, 0], [2, 2, 0], [2, 4, 0]])
cyl_radius = np.array([0.25])
# cyl_control_points = np.array([[0, 0, 0], [2, 0, 0], [2, 4, 0]])
# cyl_radius = np.array([0.25])


# Calculate the densities
densities, sample_positions, sample_radii = calculate_densities(el_centers, el_size, cyl_control_points, cyl_radius)

# Plot
plotter = pv.Plotter(shape=(1, 2), window_size=(1500, 500))

# Plot the grid without the kernel
plot_grid(plotter, (0, 0), el_centers, el_size, densities=densities)

# Plot the grid with the kernel
plot_grid(plotter, (0, 1), el_centers, el_size, densities=None)
plot_spheres(plotter, (0, 1), sample_positions, sample_radii, 'lightgray')
plot_capsules(plotter, (0, 1), cyl_control_points, cyl_radius, 'blue')

plotter.show_bounds(all_edges=True, location='outer')
plotter.show_axes()
plotter.link_views()
plotter.show()
