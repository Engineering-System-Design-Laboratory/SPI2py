import jax.numpy as jnp


def bounding_box_bounds_points(positions):

    # Find overall min and max coordinates
    x_min, y_min, z_min = jnp.min(positions, axis=0).reshape(3, 1)
    x_max, y_max, z_max = jnp.max(positions, axis=0).reshape(3, 1)

    # Combine into a single tensor representing the bounding box
    bounds = jnp.concatenate((x_min, x_max, y_min, y_max, z_min, z_max))

    return bounds


def bounding_box_bounds(positions, radii):
    """
    Calculate the bounding box that contains all spheres.

    Parameters:
    spheres (Tensor): A tensor of shape (n, 4) where each row is [x, y, z, radius]

    Returns:
    Tensor: A tensor of shape (2, 3) representing the two opposite vertices (min and max) of the bounding box.
    """

    # Calculate min and max coordinates for each sphere
    min_coords = positions - radii.reshape(-1, 1)
    max_coords = positions + radii.reshape(-1, 1)

    # Find overall min and max coordinates
    x_min, y_min, z_min = jnp.min(min_coords, axis=0).reshape(3, 1)
    x_max, y_max, z_max = jnp.max(max_coords, axis=0).reshape(3, 1)

    # Combine into a single tensor representing the bounding box
    bounds = jnp.concatenate((x_min, x_max, y_min, y_max, z_min, z_max))

    return bounds


def bounding_box_volume(bounds):

    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    return volume
