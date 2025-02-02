import numpy as np

def minimum_distance_segment_segment(a, b, c, d):
    """
    Returns the minimum Euclidean distance between two line segments. Points are zero-length line segments.

    Implementation is based on:

    Vladimir J. Lumelsky,
    "On Fast Computation of Distance Between Line Segments",
    Information Processing Letters 21 (1985) 55-61
    https://doi.org/10.1016/0020-0190(85)90032-8

    Values 0 <= t <= 1 correspond to points being inside segment AB whereas values < 0  correspond to being 'left' of AB
    and values > 1 correspond to being 'right' of AB.

    Values 0 <= u <= 1 correspond to points being inside segment CD whereas values < 0  correspond to being 'left' of CD
    and values > 1 correspond to being 'right' of CD.

    Step 1: Check for special cases; compute D1, D2, and the denominator in (11)
        (a) If one of the two segments degenerates into a point, assume that this segment corresponds to the parameter
        u, take u=0, and go to Step 4.
        (b) If both segments degenerate into points, take t=u=0, and go to Step 5.
        (c) If neither of two segments degenerates into a point and the denominator in (11) is zero, take t=0 and go to
        Step 3.
        (d) If none of (a), (b), (c) takes place, go to Step 2.
    Step 2: Using (11) compute t. If t is not in the range [0,1], modify t using (12).
    Step 3: Using (10) compute u. If u is not in the range [0,1], modify u using (12); otherwise, go to Step 5.
    Step 4: Using (10) compute t. If t is not in the range [0,1], modify t using (12).
    Step 5: With current values of t and u, compute the actual MinD using (7).

    :param a: (1,3) numpy array
    :param b: (1,3) numpy array
    :param c: (1,3) numpy array
    :param d: (1,3) numpy array

    :return: Minimum distance between line segments, float
    """

    def clamp_bound(num):
        """
        If the number is outside the range [0,1] then clamp it to the nearest boundary.
        """
        if num < 0.:
            return np.array(0.)
        elif num > 1.:
            return np.array(1.)
        else:
            return num


    d1  = b - a
    d2  = d - c
    d12 = c - a

    D1  = np.dot(d1, d1.T)
    D2  = np.dot(d2, d2.T)
    S1  = np.dot(d1, d12.T)
    S2  = np.dot(d2, d12.T)
    R   = np.dot(d1, d2.T)
    den = D1 * D2 - R**2 + 1e-8

    # Check if one or both line segments are points
    if D1 == 0. or D2 == 0.:

        # Both AB and CD are points
        if D1 == 0. and D2 == 0.:
            t = np.array(0.)
            u = np.array(0.)

        # AB is a line segment and CD is a point
        elif D1 != 0.:
            u = np.array(0.)
            t = S1/D1
            t = clamp_bound(t)

        # AB is a point and CD is a line segment
        elif D2 != 0.:
            t = np.array(0.)
            u = -S2/D2
            u = clamp_bound(u)

    # Check if line segments are parallel
    elif den == 0.:
        t = np.array(0.)
        u = -S2/D2
        uf = clamp_bound(u)

        if uf != u:
            t = (uf*R + S1)/D1
            t = clamp_bound(t)
            u = uf

    # General case for calculating the minimum distance between two line segments
    else:

        t = (S1 * D2 - S2 * R) / den

        t = clamp_bound(t)

        u = (t * R - S2) / D2
        uf = clamp_bound(u)

        if uf != u:
            t = (uf * R + S1) / D1
            t = clamp_bound(t)

            u = uf

    min_dist     = np.linalg.norm(d1 * t - d2 * u - d12)
    min_dist_pos = a + d1*t

    return min_dist, min_dist_pos