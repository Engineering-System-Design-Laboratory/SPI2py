import jax.numpy as jnp
from chex import assert_shape, assert_type
from ..geometry.spheres import get_aabb_indices
from ..geometry.intersection import volume_intersection_two_spheres
from ..projection.grid_kernels import apply_kernel
from ..utilities.aggregation import kreisselmeier_steinhauser_max, kreisselmeier_steinhauser_min

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





def calculate_pseudo_densities(grid_centers, grid_size, obj_points, obj_radii, kernel_points, kernel_radii):
    """
    Projects object points to the grid and calculates pseudo-densities.

    Parameters:
    ----------
    grid_centers : array, shape (nx, ny, nz, nk, 3)
        Grid element centers.
    grid_size : float
        Size of each grid element.
    obj_points : array, shape (no, 3)
        Object points in 3D space.
    obj_radii : array, shape (no, 1)
        Radii of the object points.
    kernel_points : array, shape (nk, 3)
        Points representing the mesh kernel.
    kernel_radii : array, shape (nk, 1)
        Radii of kernel points.

    Returns:
    --------
    all_densities : array, shape (nx, ny, nz)
        Pseudo-densities calculated for each grid element.
    kernel_points : array, shape (nxs, nys, nzs, nk, 3)
        Kernel points in the grid's active area.
    kernel_radii : array, shape (nxs, nys, nzs, nk, 1)
        Kernel radii in the grid's active area.
    """

    # Check the input shapes
    assert_shape(grid_centers, (None, None, None, None, 3))
    assert_shape(grid_size, ())
    assert_shape(obj_points, (None, 3))
    assert_shape(obj_radii, (None, 1))
    assert_shape(kernel_points, (None, 3))
    assert_shape(kernel_radii, (None, 1))

    # Check the input types
    assert_type(grid_centers, "float64")
    assert_type(grid_size, "float64")
    assert_type(obj_points, "float64")
    assert_type(obj_radii, "float64")
    assert_type(kernel_points, "float64")
    assert_type(kernel_radii, "float64")

    # Unpack the AABB indices
    i1, i2, j1, j2, k1, k2 = get_aabb_indices(grid_centers, grid_size, obj_points, obj_radii)

    # Extract grid dimensions
    grid_nx, grid_ny, grid_nz, _, _ = grid_centers.shape
    aabb_nx, aabb_ny, aabb_nz = (i2 - i1 + 1), (j2 - j1 + 1), (k2 - k1 + 1)
    obj_count, _ = obj_points.shape
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

    # Expand dimensions for broadcasting
    # Transpose object radii for broadcasting
    obj_points_bc = obj_points.reshape(1, 1, 1, 1, obj_count, 3)
    obj_radii_bc = obj_radii.T.reshape(1, 1, 1, 1, obj_count)
    kernel_points_bc = kernel_points.reshape(aabb_nx, aabb_ny, aabb_nz, kernel_count, 1, 3)

    # Compute distances between kernel and object points
    distances = jnp.linalg.norm(kernel_points_bc - obj_points_bc, axis=-1)

    # Calculate volume overlaps
    overlaps = volume_intersection_two_spheres(obj_radii_bc, kernel_radii, distances)
    element_overlaps = jnp.sum(overlaps, axis=4, keepdims=True)

    # Calculate volume fractions
    volume_fractions = (element_overlaps / element_volumes).reshape(aabb_nx, aabb_ny, aabb_nz, kernel_count)

    # Sum fractions to compute pseudo-densities
    densities = jnp.sum(volume_fractions, axis=3, keepdims=True)

    # densities = kreisselmeier_steinhauser_max(densities, axis=4)

    densities = densities.squeeze(3)

    # Store the densities in the output array
    all_densities = all_densities.at[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1].set(densities)

    return all_densities, kernel_points, kernel_radii
