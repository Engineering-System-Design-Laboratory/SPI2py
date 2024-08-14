"""Distance calculations

Provides functions to calculate the distance between classes in various ways.
"""

import jax.numpy as jnp

def distance_points(positions):

    # Calculate the center-to-center distances
    diff = positions[:, jnp.newaxis, :] - positions[jnp.newaxis, :, :]
    center_to_center_distances = jnp.sqrt(jnp.sum(diff ** 2, axis=2))

    return center_to_center_distances


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



def signed_distances_spheres_spheres(centers_a: jnp.ndarray,
                                     radii_a:   jnp.ndarray,
                                     centers_b: jnp.ndarray,
                                     radii_b:   jnp.ndarray) -> jnp.ndarray:

    delta_positions = distances_points_points(centers_a, centers_b)
    delta_radii     = sum_radii(radii_a, radii_b)

    signed_distances = delta_radii - delta_positions

    return signed_distances


# def minimum_distance_segment_segment(a: jnp.ndarray,
#                                      b: jnp.ndarray,
#                                      c: jnp.ndarray,
#                                      d: jnp.ndarray) -> tuple[float, jnp.ndarray]:
#     """
#     Returns the minimum Euclidean distance between two line segments.
#
#     This function also works for calculating the distance between a line segment and a point and a point and point.
#
#     Based on the algorithm described in:
#
#     Vladimir J. Lumelsky,
#     "On Fast Computation of Distance Between Line Segments",
#     Information Processing Letters 21 (1985) 55-61
#     https://doi.org/10.1016/0020-0190(85)90032-8
#
#     Values 0 <= t <= 1 correspond to points being inside segment AB whereas values < 0  correspond to being 'left' of AB
#     and values > 1 correspond to being 'right' of AB.
#
#     Values 0 <= u <= 1 correspond to points being inside segment CD whereas values < 0  correspond to being 'left' of CD
#     and values > 1 correspond to being 'right' of CD.
#
#     Step 1: Check for special cases; compute D1, D2, and the denominator in (11)
#         (a) If one of the two segments degenerates into a point, assume that this segment corresponds to the parameter
#         u, take u=0, and go to Step 4.
#         (b) If both segments degenerate into points, take t=u=0, and go to Step 5.
#         (c) If neither of two segments degenerates into a point and the denominator in (11) is zero, take t=0 and go to
#         Step 3.
#         (d) If none of (a), (b), (c) takes place, go to Step 2.
#     Step 2: Using (11) compute t. If t is not in the range [0,1], modify t using (12).
#     Step 3: Using (10) compute u. If u is not in the range [0,1], modify u using (12); otherwise, go to Step 5.
#     Step 4: Using (10) compute t. If t is not in the range [0,1], modify t using (12).
#     Step 5: With current values of t and u, compute the actual MinD using (7).
#
#     :param a: (1,3) numpy array
#     :param b: (1,3) numpy array
#     :param c: (1,3) numpy array
#     :param d: (1,3) numpy array
#
#     :return: Minimum distance between line segments, float
#     """
#
#     def clamp_bound(num):
#         """
#         If the number is outside the range [0,1] then clamp it to the nearest boundary.
#         """
#         if num < 0.:
#             return jnp.array(0.)
#         elif num > 1.:
#             return jnp.array(1.)
#         else:
#             return num
#
#
#     d1  = b - a
#     d2  = d - c
#     d12 = c - a
#
#     D1  = jnp.dot(d1, d1.T)
#     D2  = jnp.dot(d2, d2.T)
#     S1  = jnp.dot(d1, d12.T)
#     S2  = jnp.dot(d2, d12.T)
#     R   = jnp.dot(d1, d2.T)
#     den = D1 * D2 - R**2 + 1e-8
#
#     # Check if one or both line segments are points
#     if D1 == 0. or D2 == 0.:
#
#         # Both AB and CD are points
#         if D1 == 0. and D2 == 0.:
#             t = jnp.array(0.)
#             u = jnp.array(0.)
#
#         # AB is a line segment and CD is a point
#         elif D1 != 0.:
#             u = jnp.array(0.)
#             t = S1/D1
#             t = clamp_bound(t)
#
#         # AB is a point and CD is a line segment
#         elif D2 != 0.:
#             t = jnp.array(0.)
#             u = -S2/D2
#             u = clamp_bound(u)
#
#     # Check if line segments are parallel
#     elif den == 0.:
#         t = jnp.array(0.)
#         u = -S2/D2
#         uf = clamp_bound(u)
#
#         if uf != u:
#             t = (uf*R + S1)/D1
#             t = clamp_bound(t)
#             u = uf
#
#     # General case for calculating the minimum distance between two line segments
#     else:
#
#         t = (S1 * D2 - S2 * R) / den
#
#         t = clamp_bound(t)
#
#         u = (t * R - S2) / D2
#         uf = clamp_bound(u)
#
#         if uf != u:
#             t = (uf * R + S1) / D1
#             t = clamp_bound(t)
#
#             u = uf
#
#     minimum_distance          = jnp.linalg.norm(d1*t - d2*u - d12)
#     # minimum_distance_position = a + d1*t
#
#     return minimum_distance



def minimum_distance_segment_segment(a, b, c, d):
    """
    Vectorized calculation of minimum distances between line segments or points.
    """

    def clamp_bound(num):
        """
        Vectorized clamping for JAX arrays.
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
