import numpy as np
import pyvista as pv
from SPI2py.models.computational_geometry.distance_calculations_vectorized import minimum_distance_segment_segment


def plot_density_grid(grid_centers, spacing,
                      sphere_positions, sphere_radii,
                      cylinder_start_positions, cylinder_stop_positions, cylinder_radii,
                      densities):

    n, m, o = sphere_positions.shape[0], sphere_positions.shape[1], sphere_positions.shape[2]
    p = sphere_positions.shape[3]

    plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 500))

    # Plot 1: The grid of spheres with uniform color
    plotter.subplot(0, 0)

    for ni in range(n):
        for mi in range(m):
            for oi in range(o):
                for pi in range(p):
                    pos = sphere_positions[ni, mi, oi, pi]
                    radius = sphere_radii[ni, mi, oi, pi]
                    sphere = pv.Sphere(center=pos, radius=radius)
                    plotter.add_mesh(sphere, color='lightgray', opacity=0.25)


    plotter.add_text("Grid Spheres", position='upper_edge', font_size=14)
    plotter.show_bounds(all_edges=True)


    # Plot 2: The pipe segments
    plotter.subplot(0, 1)
    for cylinder_start_position, cylinder_stop_position, cylinder_radius in zip(cylinder_start_positions, cylinder_stop_positions, cylinder_radii):
        length = np.linalg.norm(cylinder_stop_position - cylinder_start_position)
        direction = (cylinder_stop_position - cylinder_start_position) / length
        center = (cylinder_start_position + cylinder_stop_position) / 2
        cylinder = pv.Cylinder(center=center, direction=direction, radius=cylinder_radius, height=length)
        plotter.add_mesh(cylinder, color='blue', opacity=0.7)
    plotter.add_text("Pipe Segments", position='upper_edge', font_size=14)
    plotter.show_bounds(all_edges=True)


    # Plot 3: The combined density with colored spheres
    plotter.subplot(0, 2)

    for ni in range(n):
        for mi in range(m):
            for oi in range(o):
                pos = grid_centers[ni, mi, oi]
                box = pv.Cube(center=pos, x_length=spacing, y_length=spacing, z_length=spacing)
                opacity = densities[ni, mi, oi]
                plotter.add_mesh(box, color='black', opacity=opacity)
                # sphere = pv.Sphere(center=pos, radius=radius)
                # opacity = max(0.03, min(density, 1))
                # plotter.add_mesh(sphere, color='black', opacity=opacity)

    for cylinder_start_position, cylinder_stop_position, cylinder_radius in zip(cylinder_start_positions, cylinder_stop_positions, cylinder_radii):
        length = np.linalg.norm(cylinder_stop_position - cylinder_start_position)
        direction = (cylinder_stop_position - cylinder_start_position) / length
        center = (cylinder_start_position + cylinder_stop_position) / 2
        cylinder = pv.Cylinder(center=center, direction=direction, radius=cylinder_radius, height=length)
        plotter.add_mesh(cylinder, color='blue', opacity=0.1)

    plotter.add_text("Combined Density", position='upper_edge', font_size=14)
    plotter.show_bounds(all_edges=True)

    plotter.link_views()
    plotter.view_isometric()
    # plotter.view_xy()

    plotter.show()


def create_grid(m, n, p, spacing=1.0):
    x = np.linspace(0, (m - 1) * spacing, m)
    y = np.linspace(0, (n - 1) * spacing, n)
    z = np.linspace(0, (p - 1) * spacing, p)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    pos = np.stack((xv, yv, zv), axis=-1)
    rad = np.ones((m, n, p)) * spacing / 2
    return pos, rad

def create_cylinders(points, radius):
    x1 = np.array(points[:-1])  # Start positions (-1, 3)
    x2 = np.array(points[1:])   # Stop positions (-1, 3)
    r  = np.full((x1.shape[0], 1), radius)
    return x1, x2, r


def signed_distance(x, x1, x2, r_b):

    # Convert output from JAX.numpy to numpy
    d_be = np.array(minimum_distance_segment_segment(x, x, x1, x2))

    phi_b = r_b - d_be

    return phi_b


def regularized_Heaviside(x):
    H_tilde = 0.5 + 0.75 * x - 0.25 * x ** 3  # EQ 3 in 3D
    return H_tilde


def density(phi_b, r):
    ratio = phi_b / r
    rho = np.where(ratio < -1, 0,
                   np.where(ratio > 1, 1,
                            regularized_Heaviside(ratio)))
    return rho


def calculate_densities(positions, radii, x1, x2, r):

    # Expand dimensions to allow broadcasting
    positions_expanded = positions[..., np.newaxis, :]  # shape becomes (n, m, o, p, 1, 3)
    x1_expanded = x1[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, q, 3)
    x2_expanded = x2[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, q, 3)
    r_T_expanded = r.T[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, q, 1)

    # Vectorized signed distance and density calculations using your distance function
    phi = signed_distance(positions_expanded, x1_expanded, x2_expanded, r_T_expanded)

    rho = density(phi, radii)

    # Sum densities across all cylinders

    # Combine the pseudo densities for all cylinders in each kernel sphere
    # Collapse the last axis to get the combined density for each kernel sphere
    combined_density = np.sum(rho, axis=4, keepdims=False)

    # Combine the pseudo densities for all kernel spheres in one grid
    combined_density = np.sum(combined_density, axis=3, keepdims=False)

    # Clip
    combined_density = np.clip(combined_density, 0, 1)

    return combined_density


# Create grid
n, m, o = 4, 4, 2
positions, radii = create_grid(n, m, o, spacing=1)

kernel_pos = np.array([[-0.25, -0.25, -0.25],
     [-0.25, 0.25, -0.25],
     [0.25, -0.25, -0.25],
     [0.25, 0.25, -0.25],
     [-0.25, -0.25, 0.25],
     [-0.25, 0.25, 0.25],
     [0.25, -0.25, 0.25],
     [0.25, 0.25, 0.25]])

kernel_rad = np.array([[0.25],
                       [0.25],
                       [0.25],
                       [0.25],
                       [0.25],
                       [0.25],
                       [0.25],
                       [0.25]])

def apply_kernel(grid_pos, grid_rad, kernel_pos, kernel_rad):

    # Scale factor
    element_length = grid_pos[1, 0, 0, 0] - grid_pos[0, 0, 0, 0]
    kernel_pos = element_length * kernel_pos
    kernel_rad = element_length * kernel_rad

    # Expand the positions
    grid_pos_expanded = np.expand_dims(grid_pos, axis=3)
    grid_rad_expanded = np.expand_dims(grid_rad, axis=(3, 4))
    kernel_pos_expanded = np.expand_dims(kernel_pos, axis=0)
    # kernel_rad_expanded = np.expand_dims(kernel_rad, axis=(0, 1))

    # Apply the kernels
    all_pos = grid_pos_expanded + kernel_pos_expanded
    all_rad = np.zeros_like(grid_rad_expanded) + kernel_rad

    return all_pos, all_rad

positions_expanded, radii_expanded = apply_kernel(positions, radii, kernel_pos, kernel_rad)

# positions = positions + kernel_pos
# radii = np.zeros_like(radii) + kernel_rad

# # Create line segment arrays
# line_segment_points = [(0, 0, 0), (2, 2, 0), (2, 4, 0)]
# line_segment_radius = 0.25
# X1, X2, R = create_cylinders(line_segment_points, line_segment_radius)

# Start positions
X1 = np.array([[0.5, 0.5, 0.5],
               [3.5, 0.5, 0.5]])

# Stop positions
X2 = np.array([[3.5, 0.5, 0.5],
               [3.5, 1.5, 1.5]])

# Radii
R = np.array([[0.5],
              [0.5]])

densities = calculate_densities(positions_expanded, radii_expanded, X1, X2, R)


plot_density_grid(positions, 1, positions_expanded, radii_expanded, X1, X2, R, densities)







# a = np.array([[0., 0., 0.]])
# b = np.array([[1., 0., 0.]])
# c = np.array([[0., 0., 2.]])
# d = np.array([[1., 0., 1.]])
# dist = minimum_distance_segment_segment(a, b, c, d)

# n, m, o, p, q = 2, 2, 2, 4, 5  # example dimensions
# points = np.random.rand(n, m, o, p, 3)
# start = np.random.rand(q, 3)
# stop = np.random.rand(q, 3)
#
# # Reshape arrays to allow broadcasting
# points_expanded = points[..., np.newaxis, :]  # shape becomes (n, m, o, p, 1, 3)
# start_expanded = start[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, 1, q, 3)
# stop_expanded = stop[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, 1, q, 3)
#
# distances = minimum_distance_segment_segment(points_expanded, points_expanded, start_expanded, stop_expanded)






