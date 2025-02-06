"""Distance calculations

Provides functions to calculate the distance between classes in various ways.
"""

import jax.numpy as jnp
from chex import assert_shape, assert_type


def distances_points_points(a: jnp.ndarray,
                            b: jnp.ndarray) -> jnp.ndarray:

    # Validate the inputs
    assert_shape(a, (None, 3))
    assert_shape(b, (None, 3))
    assert_type(a, 'float64')
    assert_type(b, 'float64')

    # Reshape the arrays for broadcasting
    aa = a.reshape(-1, 1, 3)
    bb = b.reshape(1, -1, 3)
    cc = aa - bb

    # Calculate the distances
    c = jnp.linalg.norm(cc, axis=2)

    return c


def sum_radii(a: jnp.ndarray,
              b: jnp.ndarray) -> jnp.ndarray:

    # Validate the inputs
    assert_shape(a, (None, 1))
    assert_shape(b, (None, 1))
    assert_type(a, 'float64')
    assert_type(b, 'float64')

    # Reshape the arrays for broadcasting
    aa = a.reshape(-1, 1)
    bb = b.reshape(1, -1)

    # Calculate the sum of the radii
    c = aa + bb

    return c


def signed_distances_spheres_spheres(centers_a: jnp.ndarray,
                                     radii_a:   jnp.ndarray,
                                     centers_b: jnp.ndarray,
                                     radii_b:   jnp.ndarray) -> jnp.ndarray:

    """
    Calculate the signed distances between two sets of spheres.

    Note: The signed distance is positive if the spheres are separated, and negative if they are intersecting.

    Parameters:
    - centers_a: The centers of the first set of spheres.
    - radii_a: The radii of the first set of spheres.
    - centers_b: The centers of the second set of spheres.
    - radii_b: The radii of the second set of spheres.

    Returns:
    - The signed distances between the two sets of spheres
    """

    # Validate the inputs
    assert_shape(centers_a, (None, 3))
    assert_shape(radii_a, (None, 1))
    assert_shape(centers_b, (None, 3))
    assert_shape(radii_b, (None, 1))
    assert_type(centers_a, 'float64')
    assert_type(radii_a, 'float64')
    assert_type(centers_b, 'float64')
    assert_type(radii_b, 'float64')

    # Calculate the signed distances
    delta_positions = distances_points_points(centers_a, centers_b)
    delta_radii     = sum_radii(radii_a, radii_b)
    signed_distances = delta_radii - delta_positions

    return signed_distances


def min_dist_segment_segment(start_1: jnp.ndarray,
                             stop_1: jnp.ndarray,
                             start_2: jnp.ndarray,
                             stop_2: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the minimum distances between line segments.

    Note 1: This function also works for points, where you set start==stop.

    Note 2: This a vectorized implementation based on https://doi.org/10.1016/0020-0190(85)90032-8.

    Parameters:
    - start_1: The starting points of the first set of line segments.
    - stop_1: The stopping points of the first set of line segments.
    - start_2: The starting points of the second set of line segments.
    - stop_2: The stopping points of the second set of line segments.

    Returns:
    - The minimum distances between the two sets of line segments.

    """

    def clamp_bound(num):
        """
        If the number is outside the range [0,1] then clamp it to the nearest boundary.
        """
        return jnp.clip(num, 0., 1.)

    # Validate the inputs
    assert_shape(start_1, (..., 3))
    assert_shape(stop_1, (..., 3))
    assert_shape(start_2, (..., 3))
    assert_shape(stop_2, (..., 3))
    assert_type(start_1, 'float64')
    assert_type(stop_1, 'float64')
    assert_type(start_2, 'float64')
    assert_type(stop_2, 'float64')

    d1 = stop_1 - start_1
    d2 = stop_2 - start_2
    d12 = start_2 - start_1

    D1 = jnp.sum(d1 * d1, axis=-1, keepdims=True)
    D2 = jnp.sum(d2 * d2, axis=-1, keepdims=True)
    S1 = jnp.sum(d1 * d12, axis=-1, keepdims=True)
    S2 = jnp.sum(d2 * d12, axis=-1, keepdims=True)
    R = jnp.sum(d1 * d2, axis=-1, keepdims=True)
    den = D1 * D2 - R ** 2 + 1e-8

    t = jnp.zeros_like(D1)
    u = jnp.zeros_like(D2)

    # Handling cases where segments degenerate into points
    mask_D1_zero = D1 == 0.
    mask_D2_zero = D2 == 0.
    mask_den_zero = den == 0.

    # Both segments are points
    mask_both_points = mask_D1_zero & mask_D2_zero

    # Segment CD is a point
    mask_CD_point = ~mask_D1_zero & mask_D2_zero
    t = jnp.where(mask_CD_point, S1 / D1, t)

    # Segment AB is a point
    mask_AB_point = mask_D1_zero & ~mask_D2_zero
    u = jnp.where(mask_AB_point, -S2 / D2, u)

    # Line segments are parallel
    u = jnp.where(mask_den_zero, -S2 / D2, u)
    uf = clamp_bound(u)
    t = jnp.where(mask_den_zero, (uf * R + S1) / D1, t)
    u = jnp.where(mask_den_zero, uf, u)

    # General case
    mask_general = ~mask_both_points & ~mask_AB_point & ~mask_CD_point & ~mask_den_zero
    t_general = (S1 * D2 - S2 * R) / den
    u_general = (t_general * R - S2) / D2
    t = jnp.where(mask_general, clamp_bound(t_general), t)
    u = jnp.where(mask_general, clamp_bound(u_general), u)
    u = clamp_bound(u)
    t = jnp.where(mask_general, clamp_bound((u * R + S1) / D1), t)

    minimum_distance = jnp.linalg.norm(d1 * t - d2 * u - d12, axis=-1)

    return minimum_distance


def min_dist_point_segment(point: jnp.ndarray,
                           start: jnp.ndarray,
                           stop: jnp.ndarray) -> jnp.ndarray:

    # Validate the inputs
    assert_shape(point, (None, 3))
    assert_shape(start, (None, 3))
    assert_shape(stop, (None, 3))
    assert_type(point, 'float64')
    assert_type(start, 'float64')
    assert_type(stop, 'float64')

    min_dist = min_dist_segment_segment(point, point, start, stop)

    return min_dist


def signed_distances_capsule_capsule(centers_1: jnp.ndarray,
                                     radii_1:   jnp.ndarray,
                                     centers_2: jnp.ndarray,
                                     radii_2:   jnp.ndarray) -> jnp.ndarray:

    # Validate the inputs
    assert_shape(centers_1, (None, 3))
    assert_shape(radii_1, (None, 1))
    assert_shape(centers_2, (None, 3))
    assert_shape(radii_2, (None, 1))
    assert_type(centers_1, 'float64')
    assert_type(radii_1, 'float64')
    assert_type(centers_2, 'float64')
    assert_type(radii_2, 'float64')

    delta_positions = min_dist_segment_segment(centers_1, centers_2)
    delta_radii     = sum_radii(radii_1, radii_2)

    signed_distances = delta_radii - delta_positions

    return signed_distances
