import numpy as np
import pyvista as pv

def plot(grid_pos, grid_rad,
         bar_start_pos, bar_stop_pos, bar_rad,
         rho,
         title):

    # Get the dimensions of the grid
    m, n, p = grid_pos.shape[0], grid_pos.shape[1], grid_pos.shape[2]

    # Initialize the plotter
    plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 500))

    # Add the title
    plotter.add_text(title, position='upper_edge', font_size=14)

    # Plot 1: The grid
    plotter.subplot(0, 0)
    plotter.add_text("Grid", position='upper_edge', font_size=14)
    plotter.show_bounds(all_edges=True)
    for mi in range(m):
        for ni in range(n):
            for pi in range(p):
                pos = grid_pos[mi, ni, pi]
                radius = grid_rad[mi, ni, pi]
                sphere = pv.Sphere(center=pos, radius=radius)
                plotter.add_mesh(sphere, color='lightgray', opacity=0.25)

    # Plot 2: The bars
    plotter.subplot(0, 1)
    plotter.add_text("Bars", position='upper_edge', font_size=14)
    plotter.show_bounds(all_edges=True)
    for bar_start_posi, bar_stop_posi, bar_radi in zip(bar_start_pos, bar_stop_pos, bar_rad):
        length = np.linalg.norm(bar_stop_posi - bar_start_posi)
        direction = (bar_stop_posi - bar_start_posi) / length
        center = (bar_start_posi + bar_stop_posi) / 2
        cylinder = pv.Cylinder(center=center, direction=direction, radius=bar_radi, height=length)
        plotter.add_mesh(cylinder, color='blue', opacity=0.7)

    # Plot 3: The combined density with colored spheres
    plotter.subplot(0, 2)
    plotter.add_text("Projection with Bars Overlaid", position='upper_edge', font_size=14)
    plotter.show_bounds(all_edges=True)

    # Plot the grid
    for mi in range(m):
        for ni in range(n):
            for pi in range(p):
                pos = grid_pos[mi, ni, pi]
                radius = grid_rad[mi, ni, pi]
                density = rho[mi, ni, pi]
                sphere = pv.Sphere(center=pos, radius=radius)
                opacity = max(0, min(density, 1))
                plotter.add_mesh(sphere, color='black', opacity=opacity)

    # Plot the bars
    for bar_start_posi, bar_stop_posi, bar_radi in zip(bar_start_pos, bar_stop_pos, bar_rad):
        length = np.linalg.norm(bar_stop_posi - bar_start_posi)
        direction = (bar_stop_posi - bar_start_posi) / length
        center = (bar_start_posi + bar_stop_posi) / 2
        cylinder = pv.Cylinder(center=center, direction=direction, radius=bar_radi, height=length)
        plotter.add_mesh(cylinder, color='blue', opacity=0.1)

    # Configure the plot
    plotter.link_views()
    plotter.view_isometric()
    plotter.show()


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
        These calculations are for line segments, not lines of infinite length.
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


def create_grid(m, n, p, spacing=1.0):
    x = np.linspace(0, (m - 1) * spacing, m)
    y = np.linspace(0, (n - 1) * spacing, n)
    z = np.linspace(0, (p - 1) * spacing, p)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    pos = np.stack((xv, yv, zv), axis=-1)
    rad = np.ones((m, n, p)) * spacing / 2
    return pos, rad

def d_b(x_e, x_1b, x_2b):
    x_2b1b = x_2b - x_1b  # EQ 9
    x_e1b = x_e - x_1b  # EQ 9
    x_e2b = x_e - x_2b  # EQ 9
    l_b = np.linalg.norm(x_2b1b)  # EQ 10
    a_b = x_2b1b / l_b  # EQ 11
    l_be = np.dot(a_b, x_e1b)  # 12  TODO Dot or mult?
    r_be = np.linalg.norm(x_e1b - (l_be * a_b))  # EQ 13

    # EQ 14
    if l_be <= 0:
        return np.linalg.norm(x_e1b)
    elif l_be > l_b:
        return np.linalg.norm(x_e2b)
    else:
        return r_be

def phi_b(x, x1, x2, r_b):

    d_be = d_b(x, x1, x2)

    # EQ 8
    phi_b = d_be - r_b

    return phi_b

def phi_b_RevSign(x, x1, x2, r_b):

    d_be = d_b(x, x1, x2)

    # EQ 8 (order reversed)
    phi_b = r_b - d_be

    return phi_b


def phi_b_AltDist(x, x1, x2, r_b):
    d_be, _ = minimum_distance_segment_segment(x, x, x1, x2)

    d_be = float(d_be)

    # EQ 8
    phi_b = d_be - r_b

    return phi_b

def phi_b_RevSign_AltDist(x, x1, x2, r_b):

    d_be, _ = minimum_distance_segment_segment(x, x, x1, x2)

    d_be = float(d_be)

    # EQ 8 (order reversed)
    phi_b = r_b - d_be

    return phi_b


def H_tilde(x):
    # EQ 3 in 3D
    return 0.5 + 0.75 * x - 0.25 * x ** 3


def rho_b(phi_b, r):
    # EQ 2
    if phi_b / r < -1:
        return 0

    elif -1 <= phi_b / r <= 1:
        return H_tilde(phi_b / r)

    elif phi_b / r > 1:
        return 1

    else:
        raise ValueError('Something went wrong')


def calculate_density(position, radius, x1, x2, r, mode):

    if mode == 'phi_b':
        phi = phi_b(position, x1, x2, r)
    elif mode == 'phi_b_RevSign':
        phi = phi_b_RevSign(position, x1, x2, r)
    elif mode == 'phi_b_AltDist':
        phi = phi_b_AltDist(position, x1, x2, r)
    elif mode == 'phi_b_RevSign_AltDist':
        phi = phi_b_RevSign_AltDist(position, x1, x2, r)

    rho = rho_b(phi, radius)
    return rho

def calculate_combined_density(position, radius, X1, X2, R, mode):

    combined_density = 0
    for x1, x2, r in zip(X1, X2, R):
        density = calculate_density(position, radius, x1, x2, r, mode)
        combined_density += density

    # Clip the combined densities to be between 0 and 1
    combined_density = max(0, min(combined_density, 1))

    return combined_density

def calculate_combined_densities(positions, radii, X1, X2, R, mode):
    m, n, p = positions.shape[0], positions.shape[1], positions.shape[2]
    densities = np.zeros((m, n, p))
    for i in range(m):
        for j in range(n):
            for k in range(p):
                position = positions[i, j, k]
                radius = radii[i, j, k]
                combined_density = calculate_combined_density(position, radius, X1, X2, R, mode)
                densities[i, j, k] += combined_density

    return densities


grid_pos, grid_rad = create_grid(7, 7, 4, spacing=1)
X1 = np.array([[0, 0, 0], [2, 2, 0]])
X2 = np.array([[2, 2, 0], [2, 4, 0]])
R = np.array([[0.25], [0.25]])


rho1 = calculate_combined_densities(grid_pos, grid_rad, X1, X2, R, mode='phi_b')
rho2 = calculate_combined_densities(grid_pos, grid_rad, X1, X2, R, mode='phi_b_RevSign')
rho3 = calculate_combined_densities(grid_pos, grid_rad, X1, X2, R, mode='phi_b_AltDist')
rho4 = calculate_combined_densities(grid_pos, grid_rad, X1, X2, R, mode='phi_b_RevSign_AltDist')

plot(grid_pos, grid_rad, X1, X2, R, rho1, title='phi_b')
plot(grid_pos, grid_rad, X1, X2, R, rho2, title='phi_b_RevSign')
plot(grid_pos, grid_rad, X1, X2, R, rho3, title='phi_b_AltDist')
plot(grid_pos, grid_rad, X1, X2, R, rho4, title='phi_b_RevSign_AltDist')


