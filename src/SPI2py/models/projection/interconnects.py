"""
From Norato's paper...
"""

import numpy as np
from ..geometry.cylinders import create_cylinders
from ..mechanics.distance import minimum_distance_segment_segment
from ..projection.kernels_uniform import apply_kernel

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


def calculate_combined_densities(positions, radii, x1, x2, r):

    # Expand dimensions to allow broadcasting
    positions_expanded = positions[..., np.newaxis, :]  # shape becomes (n, m, o, p, 1, 3)
    x1_expanded = x1[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, q, 3)
    x2_expanded = x2[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, q, 3)
    r_T_expanded = r.T[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, q, 1)

    # Vectorized signed distance and density calculations using your distance function
    phi = signed_distance(positions_expanded, x1_expanded, x2_expanded, r_T_expanded)

    rho = density(phi, radii)

    # Sum densities across all cylinders
    # TODO Sanity check multiple spheres... sum 4,5
    # combined_density = np.clip(np.sum(rho, axis=4), 0, 1)

    # Combine the pseudo densities for all cylinders in each kernel sphere
    # Collapse the last axis to get the combined density for each kernel sphere
    combined_density = np.sum(rho, axis=4, keepdims=False)

    # Combine the pseudo densities for all kernel spheres in one grid
    combined_density = np.sum(combined_density, axis=3, keepdims=False)

    # Clip
    combined_density = np.clip(combined_density, 0, 1)

    return combined_density

def calculate_densities(el_centers, el_size,
                        cyl_control_points, cyl_radius, kernel_pos, kernel_rad):
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