"""
From Norato's paper...
"""

import numpy as np

def d_b(x_e, x_1b, x_2b):
    x_2b1b = x_2b - x_1b  # EQ 9
    x_e1b = x_e - x_1b  # EQ 9
    x_e2b = x_e - x_2b  # EQ 9
    l_b = np.linalg.norm(x_2b1b)  # EQ 10
    a_b = x_2b1b / l_b  # EQ 11
    l_be = np.dot(a_b, x_e1b)  # 12
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


def calculate_density(position, radius, x1, x2, r):
    phi = phi_b(position, x1, x2, r)
    rho = rho_b(phi, radius)
    return rho

def calculate_combined_density(position, radius, X1, X2, R):

    combined_density = 0
    for x1, x2, r in zip(X1, X2, R):
        density = calculate_density(position, radius, x1, x2, r)
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