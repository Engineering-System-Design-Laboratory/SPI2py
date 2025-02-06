import jax.numpy as jnp
from chex import assert_shape, assert_type


def volume_intersection_two_spheres(radii_1: jnp.ndarray,
                                    radii_2: jnp.ndarray,
                                    distances: jnp.ndarray) -> jnp.ndarray:

    # Validate the inputs
    assert_shape(radii_1, (None, 1))
    assert_shape(radii_2, (None, 1))
    assert_shape(distances, (None, 1))
    assert_type(radii_1, 'float64')
    assert_type(radii_2, 'float64')
    assert_type(distances, 'float64')

    # Calculate volumes for all spheres
    volume_1 = (4 / 3) * jnp.pi * radii_1 ** 3
    volume_2 = (4 / 3) * jnp.pi * radii_2 ** 3

    # Calculate intersection volume for all pairs, assuming overlapping but not fully enclosed
    numerator = jnp.pi * (radii_1 + radii_2 - distances) ** 2 * (
            distances ** 2 + 2 * distances * radii_2 - 3 * radii_2 ** 2 + 2 * distances * radii_1 + 6 * radii_2 * radii_1 - 3 * radii_1 ** 2)
    denominator = 12 * distances
    intersection_volume = jnp.where(denominator != 0, numerator / denominator, jnp.zeros_like(distances))

    # Condition for when one sphere is fully within another (not touching boundary)
    fully_inside = distances + jnp.minimum(radii_1, radii_2) <= jnp.maximum(radii_1, radii_2)

    # When one sphere is fully inside another, use the volume of the smaller sphere
    overlap_volume = jnp.where(fully_inside, jnp.minimum(volume_1, volume_2), intersection_volume)

    # Condition for no overlap (d >= r_1 + r_2)
    no_overlap = distances >= (radii_1 + radii_2)
    overlap_volume = jnp.where(no_overlap, jnp.zeros_like(distances), overlap_volume)

    return overlap_volume


def total_overlap_volume(centers: jnp.ndarray,
                         radii: jnp.ndarray) -> jnp.ndarray:

    # Validate the inputs
    assert_shape(centers, (None, 3))
    assert_shape(radii, (None, 1))
    assert_type(centers, 'float64')
    assert_type(radii, 'float64')

    # Reshape the arrays for broadcasting
    centers_a = centers.reshape(-1, 1, 3)
    centers_b = centers.reshape(1, -1, 3)
    radii_a = radii.reshape(-1, 1)
    radii_b = radii.reshape(1, -1)

    # Calculate the distances between all pairs of spheres
    dist = jnp.linalg.norm(centers_a - centers_b, axis=2)

    # Get the upper triangular indices
    triu_indices = jnp.triu_indices(len(centers_a), k=1)

    dist = dist[triu_indices]
    radii_a = radii_a.reshape(-1)[triu_indices[0]]
    radii_b = radii_b.reshape(-1)[triu_indices[1]]

    total_overlap = volume_intersection_two_spheres(radii_a, radii_b, dist).sum()

    return total_overlap


def volume_sphere_cap():
    raise NotImplementedError
