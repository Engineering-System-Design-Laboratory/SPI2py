import jax.numpy as jnp
from SPI2py.models.geometry.encapsulation import overlap_volume_sphere_sphere

def calculate_pseudo_densities(sphere_positions, sphere_radii, sample_points, sample_radii, element_bounds):
    """
    Projects the points to the mesh and calculates the pseudo-densities

    mesh_positions: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1, 3) tensor
    mesh_radii: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1) tensor
    object_positions_expanded: (1, 1, 1, 1, n_object_points, 3) tensor
    object_radii_expanded: (1, 1, 1, 1, n_object_points) tensor

    pseudo_densities: (n_el_x, n_el_y, n_el_z, 1) tensor
    """

    element_sample_volumes = (4 / 3) * jnp.pi * sample_radii ** 3
    element_volumes = jnp.sum(element_sample_volumes, axis=3, keepdims=True)

    object_positions = sphere_positions
    object_radii = sphere_radii
    object_radii_transposed = object_radii.T
    object_radii_expanded = object_radii_transposed[None, None, None, ...]

    # Assuming sample_points, object_positions, object_radii_expanded, sample_radii, and element_volumes are JAX arrays
    distances = jnp.linalg.norm(sample_points[..., None, :] - object_positions[None, None, None, None, :, :], axis=-1)
    volume_sample_overlaps = overlap_volume_sphere_sphere(object_radii_expanded, sample_radii, distances)
    volume_element_overlaps = jnp.sum(volume_sample_overlaps, axis=4, keepdims=True)
    volume_fractions = volume_element_overlaps / element_volumes
    pseudo_densities = jnp.sum(volume_fractions, axis=(3, 4), keepdims=True).squeeze(3)

    # Clip the pseudo-densities
    pseudo_densities = jnp.clip(pseudo_densities, a_min=0, a_max=1)

    return pseudo_densities

# def calculate_pseudo_densities(sphere_positions, sphere_radii, sample_points, sample_radii, bounds):
#     """
#     Projects the points to the mesh and calculates the pseudo-densities, considering only the mesh elements within specified bounds.
#
#     Parameters:
#     - sphere_positions: (n_object_points, 3) tensor of object sphere positions.
#     - sphere_radii: (n_object_points,) tensor of object sphere radii.
#     - sample_points: (n_el_x, n_el_y, n_el_z, n_mesh_points, 3) tensor of mesh sphere positions.
#     - sample_radii: (n_el_x, n_el_y, n_el_z, n_mesh_points) tensor of mesh sphere radii.
#     - bounds: Tuple specifying the bounds (x_min, x_max, y_min, y_max, z_min, z_max).
#
#     Returns:
#     - pseudo_densities: (n_el_x, n_el_y, n_el_z, 1) tensor of pseudo-densities.
#     """
#
#     x_min, x_max, y_min, y_max, z_min, z_max = bounds.T
#
#     # Filter sample_points based on bounds
#     in_bounds_mask = (sample_points[..., 0] >= x_min) & (sample_points[..., 0] <= x_max) & \
#                      (sample_points[..., 1] >= y_min) & (sample_points[..., 1] <= y_max) & \
#                      (sample_points[..., 2] >= z_min) & (sample_points[..., 2] <= z_max)
#
#     # Expand the mask to shape (n_el_x, n_el_y, n_el_z, n_mesh_points, 1) for broadcasting
#     in_bounds_mask_expanded = in_bounds_mask[..., None]
#
#     # Use the in_bounds_mask to filter sample_radii, setting radii to 0 for out-of-bounds points
#     sample_radii_filtered = jnp.where(in_bounds_mask_expanded, sample_radii, 0)
#
#     element_sample_volumes = (4 / 3) * jnp.pi * sample_radii_filtered ** 3
#     element_volumes = jnp.sum(element_sample_volumes, axis=3, keepdims=True)
#
#     object_radii_expanded = sphere_radii[:, None, None, None, None, None]  # Adjusted shape for broadcasting
#
#     distances = jnp.linalg.norm(sample_points[..., None, :] - sphere_positions[None, None, None, None, :, :], axis=-1)
#     volume_sample_overlaps = overlap_volume_sphere_sphere(object_radii_expanded, sample_radii_filtered, distances)
#     volume_element_overlaps = jnp.sum(volume_sample_overlaps, axis=4, keepdims=True)
#     volume_fractions = volume_element_overlaps / element_volumes
#     pseudo_densities = jnp.sum(volume_fractions, axis=(3, 4), keepdims=True).squeeze(3)
#
#     # Clip the pseudo-densities
#     pseudo_densities = jnp.clip(pseudo_densities, a_min=0, a_max=1)
#
#     return pseudo_densities

# def calculate_pseudo_densities(sphere_positions, sphere_radii, sample_points, sample_radii, aabb, element_bounds):
#     """
#     Projects the points to the mesh and calculates the pseudo-densities
#
#     TODO Implement AABB to reduce the number of calculations
#
#     mesh_positions: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1, 3) tensor
#     mesh_radii: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1) tensor
#     object_positions_expanded: (1, 1, 1, 1, n_object_points, 3) tensor
#     object_radii_expanded: (1, 1, 1, 1, n_object_points) tensor
#
#     pseudo_densities: (n_el_x, n_el_y, n_el_z, 1) tensor
#     """
#
#     element_sample_volumes = (4 / 3) * jnp.pi * sample_radii ** 3
#     element_volumes = jnp.sum(element_sample_volumes, axis=3, keepdims=True)
#
#     sample_points_expanded = sample_points[..., None, :]
#
#     object_positions = sphere_positions
#     object_positions_expanded = object_positions[None, None, None, None, :, :]
#
#     object_radii = sphere_radii
#     object_radii_transposed = object_radii.T
#     object_radii_expanded = object_radii_transposed[None, None, None, ...]
#
#     #
#     el_x_min, el_x_max = element_bounds[..., 0], element_bounds[..., 1]
#     el_y_min, el_y_max = element_bounds[..., 2], element_bounds[..., 3]
#     el_z_min, el_z_max = element_bounds[..., 4], element_bounds[..., 5]
#
#     obj_x_min, obj_x_max, obj_y_min, obj_y_max, obj_z_min, obj_z_max = aabb.T
#
#     # Identify mesh elements that are within the object's AABB
#     element_in_bounds = (el_x_min <= obj_x_max) & (el_x_max >= obj_x_min) & \
#                         (el_y_min <= obj_y_max) & (el_y_max >= obj_y_min) & \
#                         (el_z_min <= obj_z_max) & (el_z_max >= obj_z_min)
#
#     element_in_bounds_expanded = element_in_bounds[..., None]
#     relevant_indices = jnp.where(element_in_bounds_expanded)
#
#     relevant_sample_points = sample_points_expanded[element_in_bounds]
#
#     # Assuming sample_points, object_positions, object_radii_expanded, sample_radii, and element_volumes are JAX arrays
#     # distances = jnp.linalg.norm(sample_points_expanded - object_positions_expanded, axis=-1)
#     distances = jnp.linalg.norm(relevant_sample_points - object_positions_expanded, axis=-1)
#
#     # volume_sample_overlaps = overlap_volume_sphere_sphere(object_radii_expanded, sample_radii, distances)
#     volume_sample_overlaps = overlap_volume_sphere_sphere(object_radii_expanded, sample_radii[relevant_indices], distances)
#     volume_element_overlaps = jnp.sum(volume_sample_overlaps, axis=4, keepdims=True)
#     # volume_fractions = volume_element_overlaps / element_volumes
#     volume_fractions = volume_element_overlaps / element_volumes[relevant_indices]
#
#     # pseudo_densities = jnp.sum(volume_fractions, axis=(3, 4), keepdims=True).squeeze(3)
#     relevant_pseudo_densities = jnp.sum(volume_fractions, axis=(3, 4), keepdims=True).squeeze(3)
#     pseudo_densities = jnp.where(element_in_bounds, relevant_pseudo_densities.squeeze(3), 0)
#
#     # Clip the pseudo-densities
#     pseudo_densities = jnp.clip(pseudo_densities, a_min=0, a_max=1)
#
#     return pseudo_densities