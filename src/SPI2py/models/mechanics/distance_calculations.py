"""Distance calculations

Provides functions to calculate the distance between classes in various ways.
"""

import jax.numpy as jnp


def distances_points_points(a: jnp.ndarray,
                            b: jnp.ndarray) -> jnp.ndarray:

    # # Reshape the arrays for broadcasting
    aa = a.reshape(-1, 1, 3)
    bb = b.reshape(1, -1, 3)
    cc = aa-bb

    c = jnp.linalg.norm(cc, axis=2)

    return c


def sum_radii(a, b):

    aa = a.reshape(-1, 1)
    bb = b.reshape(1, -1)

    c = aa + bb

    return c



def distances_spheres_spheres(centers_a: jnp.ndarray,
                              radii_a:   jnp.ndarray,
                              centers_b: jnp.ndarray,
                              radii_b:   jnp.ndarray) -> jnp.ndarray:

    delta_positions = distances_points_points(centers_a, centers_b)
    delta_radii     = sum_radii(radii_a, radii_b)

    signed_distances = delta_radii - delta_positions

    return signed_distances


def minimum_distance_segment_segment(a, b, c, d):
    """
    Returns the minimum distances between line segments (and/or points).

    Implementation based on:

    Vladimir J. Lumelsky,
    "On Fast Computation of Distance Between Line Segments",
    Information Processing Letters 21 (1985) 55-61
    https://doi.org/10.1016/0020-0190(85)90032-8
    """

    def clamp_bound(num):
        """
        If the number is outside the range [0,1] then clamp it to the nearest boundary.
        """
        return jnp.clip(num, 0., 1.)

    d1 = b - a
    d2 = d - c
    d12 = c - a

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


def signed_distances_capsules_capsules(centers_a: jnp.ndarray,
                                       radii_a:   jnp.ndarray,
                                       centers_b: jnp.ndarray,
                                       radii_b:   jnp.ndarray) -> jnp.ndarray:

    """"""

    delta_positions = minimum_distance_segment_segment(centers_a, centers_b)
    delta_radii     = sum_radii(radii_a, radii_b)

    signed_distances = delta_radii - delta_positions

    return signed_distances


def signed_distance(x, x1, x2, r_b):

    # Convert output from JAX.numpy to numpy
    d_be = jnp.array(minimum_distance_segment_segment(x, x, x1, x2))

    phi_b = r_b - d_be

    return phi_b