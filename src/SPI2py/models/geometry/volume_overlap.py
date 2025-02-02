import jax.numpy as jnp


def volume_intersection_two_spheres(r_1, r_2, d):
    # Calculate volumes for all spheres
    volume_1 = (4 / 3) * jnp.pi * r_1 ** 3
    volume_2 = (4 / 3) * jnp.pi * r_2 ** 3

    # Calculate intersection volume for all pairs, assuming overlapping but not fully enclosed
    numerator = jnp.pi * (r_1 + r_2 - d) ** 2 * (
            d ** 2 + 2 * d * r_2 - 3 * r_2 ** 2 + 2 * d * r_1 + 6 * r_2 * r_1 - 3 * r_1 ** 2)
    denominator = 12 * d
    intersection_volume = jnp.where(denominator != 0, numerator / denominator, jnp.zeros_like(d))

    # Condition for when one sphere is fully within another (not touching boundary)
    fully_inside = d + jnp.minimum(r_1, r_2) <= jnp.maximum(r_1, r_2)

    # When one sphere is fully inside another, use the volume of the smaller sphere
    overlap_volume = jnp.where(fully_inside, jnp.minimum(volume_1, volume_2), intersection_volume)

    # Condition for no overlap (d >= r_1 + r_2)
    no_overlap = d >= (r_1 + r_2)
    overlap_volume = jnp.where(no_overlap, jnp.zeros_like(d), overlap_volume)

    return overlap_volume

def total_overlap_volume(sphere_centers, sphere_radii):

    centers_a = sphere_centers.reshape(-1, 1, 3)
    centers_b = sphere_centers.reshape(1, -1, 3)
    radii_a = sphere_radii.reshape(-1, 1)
    radii_b = sphere_radii.reshape(1, -1)

    d = jnp.linalg.norm(centers_a - centers_b, axis=2)

    # Get the upper triangular indices
    triu_indices = jnp.triu_indices(len(centers_a), k=1)

    d = d[triu_indices]
    radii_a = radii_a.reshape(-1)[triu_indices[0]]
    radii_b = radii_b.reshape(-1)[triu_indices[1]]

    total_overlap = volume_intersection_two_spheres(radii_a, radii_b, d).sum()

    return total_overlap


def volume_sphere_cap():
    pass
