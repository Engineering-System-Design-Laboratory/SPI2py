"""
Geometry Project by Norato...
"""

import jax.numpy as jnp
from ..geometry.cylinders import create_cylinders
from ..mechanics.distance import minimum_distance_segment_segment, signed_distances_capsules_capsules
from ..projection.kernels_uniform import apply_kernel
from ..geometry.spheres import get_aabb_indices


def signed_distance(x, x1, x2, r_b):

    # Convert output from JAX.numpy to numpy
    d_be = jnp.array(minimum_distance_segment_segment(x, x, x1, x2))

    # EQ 8 (order reversed, this is a confirmed typo in the paper)
    phi_b = r_b - d_be

    return phi_b


def regularized_Heaviside(x):
    H_tilde = 0.5 + 0.75 * x - 0.25 * x ** 3  # EQ 3 in 3D
    return H_tilde


def density(phi_b, r):
    ratio = phi_b / r  # EQ 2
    rho = jnp.where(ratio < -1, 0,
                    jnp.where(ratio > 1, 1,
                              regularized_Heaviside(ratio)))
    return rho

# def d_b(x_e, x_1b, x_2b):
#     """Norato"""
#     x_2b1b = x_2b - x_1b  # EQ 9
#     x_e1b = x_e - x_1b  # EQ 9
#     x_e2b = x_e - x_2b  # EQ 9
#     l_b = np.linalg.norm(x_2b1b)  # EQ 10
#     a_b = x_2b1b / l_b  # EQ 11
#     l_be = np.dot(a_b, x_e1b)  # 12  TODO Dot or mult?
#     r_be = np.linalg.norm(x_e1b - (l_be * a_b))  # EQ 13
#
#     # EQ 14
#     if l_be <= 0:
#         return np.linalg.norm(x_e1b)
#     elif l_be > l_b:
#         return np.linalg.norm(x_e2b)
#     else:
#         return r_be

# def phi_b_RevSign_AltDist(x, x1, x2, r_b):
#
#     d_be, _ = minimum_distance_segment_segment(x, x, x1, x2)
#
#     d_be = float(d_be)
#
#     # EQ 8 (order reversed)
#     phi_b = r_b - d_be
#
#     return phi_b


# def calculate_combined_densities(positions, radii, x1, x2, r):
#
#     # Expand dimensions to allow broadcasting
#     positions_expanded = positions[..., jnp.newaxis, :]  # shape becomes (n, m, o, p, 1, 3)
#     x1_expanded = x1[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :]  # shape becomes (1, 1, 1, q, 3)
#     x2_expanded = x2[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :]  # shape becomes (1, 1, 1, q, 3)
#     r_T_expanded = r.T[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :]  # shape becomes (1, 1, 1, q, 1)
#
#     # Vectorized signed distance and density calculations using your distance function
#     phi = signed_distance(positions_expanded, x1_expanded, x2_expanded, r_T_expanded)
#
#     rho = density(phi, radii)
#
#     # Sum densities across all cylinders
#     # TODO Sanity check multiple spheres... sum 4,5
#     # combined_density = np.clip(np.sum(rho, axis=4), 0, 1)
#
#     # Combine the pseudo densities for all cylinders in each kernel sphere
#     # Collapse the last axis to get the combined density for each kernel sphere
#     combined_density = jnp.sum(rho, axis=4, keepdims=False)
#
#     # Combine the pseudo densities for all kernel spheres in one grid
#     combined_density = jnp.sum(combined_density, axis=3, keepdims=False)
#
#     # Clip
#     combined_density = jnp.clip(combined_density, 0, 1)
#
#     return combined_density

def calculate_densities(grid_centers, grid_size,
                        cyl_points, cyl_radius,
                        kernel_points, kernel_radii):
    """
    Projects the points to the mesh and calculates the pseudo-densities

    mesh_positions: (n_el_x, n_el_y, n_el_z, n_mesh_points, 3) tensor
    mesh_radii: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1) tensor

    mesh_positions_expanded: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1, 3) tensor

    cylinder_starts: (n_segments, 3) tensor
    cylinder_stops: (n_segments, 3) tensor
    cylinder_radii: (n_segments, 1) tensor

    cylinder_starts_expanded: (1, 1, 1, n_segments, 3) tensor
    cylinder_stops_expanded: (1, 1, 1, n_segments, 3) tensor
    cylinder_radii_expanded: (1, 1, 1, n_segments, 1) tensor

    pseudo_densities: (n_el_x, n_el_y, n_el_z) tensor
    """

    # Create the cylinders
    cyl_starts, cyl_stops, cyl_rad = create_cylinders(cyl_points, cyl_radius)

    # Unpack the AABB indices
    i1, i2, j1, j2, k1, k2 = get_aabb_indices(grid_centers, grid_size, cyl_points, cyl_radius)

    # Extract grid dimensions
    grid_nx, grid_ny, grid_nz, _, _ = grid_centers.shape
    aabb_nx, aabb_ny, aabb_nz = (i2 - i1 + 1), (j2 - j1 + 1), (k2 - k1 + 1)
    cyl_count, _ = cyl_rad.shape
    kernel_count, _ = kernel_points.shape

    # Initialize the output density array
    all_densities = jnp.zeros((grid_nx, grid_ny, grid_nz), dtype='float64')

    # Extract the active grid region within the object's AABB
    active_grid_centers = grid_centers[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1]

    # Apply the kernel to active grid elements
    kernel_points, kernel_radii = apply_kernel(active_grid_centers, grid_size, kernel_points, kernel_radii)

    # Calculate sample volumes and element volumes
    sample_volumes = (4 / 3) * jnp.pi * kernel_radii ** 3
    element_volumes = jnp.sum(sample_volumes, axis=3, keepdims=True)

    # Expand the arrays to allow broadcasting
    # Transpose object radii for broadcasting
    cyl_starts_bc = cyl_starts.reshape(1, 1, 1, cyl_count, 3)
    cyl_stops_bc = cyl_stops.reshape(1, 1, 1, cyl_count, 3)
    cyl_rad_bc = cyl_rad.T.reshape(1, 1, 1, cyl_count, 1)
    kernel_points_bc = kernel_points.reshape(aabb_nx, aabb_ny, aabb_nz, kernel_count, 1, 3)

    # Vectorized signed distance and density calculations using your distance function
    distances = signed_distance(kernel_points_bc, cyl_starts_bc, cyl_stops_bc, cyl_rad_bc)
    # distances = signed_distances_capsules_capsules(kernel_points_bc, cyl_starts_bc, cyl_stops_bc, cyl_rad_bc)

    # Fix rho for mesh_radii?
    densities = density(distances, kernel_radii)

    # Sum densities across all cylinders
    # Combine the pseudo densities for all cylinders in each kernel sphere
    # Collapse the last axis to get the combined density for each kernel sphere
    densities = jnp.sum(densities, axis=4)

    # Combine the pseudo densities for all kernel spheres in one grid
    # densities = jnp.sum(densities, axis=3, keepdims=False)
    densities = jnp.sum(densities, axis=3)

    # Store the densities in the output array
    # TODO Why black box?
    all_densities = all_densities.at[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1].set(densities)

    # Clip
    # combined_density = np.clip(combined_density, 0, 1)

    return densities, kernel_points, kernel_radii


# def calculate_combined_density(position, radius, X1, X2, R, mode):
#
#     combined_density = 0
#     for x1, x2, r in zip(X1, X2, R):
#         density = calculate_density(position, radius, x1, x2, r, mode)
#         combined_density += density
#
#     # Clip the combined densities to be between 0 and 1
#     combined_density = max(0, min(combined_density, 1))
#
#     return combined_density

# def calculate_combined_densities(positions, radii, X1, X2, R, mode):
#     m, n, p = positions.shape[0], positions.shape[1], positions.shape[2]
#     densities = np.zeros((m, n, p))
#     for i in range(m):
#         for j in range(n):
#             for k in range(p):
#                 position = positions[i, j, k]
#                 radius = radii[i, j, k]
#                 combined_density = calculate_combined_density(position, radius, X1, X2, R, mode)
#                 densities[i, j, k] += combined_density
#
#     return densities