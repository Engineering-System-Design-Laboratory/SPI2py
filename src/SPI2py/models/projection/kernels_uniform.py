import numpy as np


def create_uniform_inscription_kernel(steps_per_edge):
    # Step size (distance between the centers of two consecutive spheres)
    step_size = 1.0 / steps_per_edge

    # Initialize arrays for sphere positions and radii
    positions = np.zeros((steps_per_edge, steps_per_edge, steps_per_edge, 3))  # 3 for (x, y, z) coordinates
    radii = np.zeros((steps_per_edge, steps_per_edge, steps_per_edge))  # Radii of spheres

    # Radius of each inscribed sphere is half of the small cube's edge length
    sphere_radius = 0.5 * step_size

    # Loop through the grid and compute the sphere positions and radii
    for i in range(steps_per_edge):
        for j in range(steps_per_edge):
            for k in range(steps_per_edge):
                # The center of each inscribed sphere is at the center of each small cube
                x = (i + 0.5) * step_size
                y = (j + 0.5) * step_size
                z = (k + 0.5) * step_size

                positions[i, j, k] = [x, y, z]
                radii[i, j, k] = sphere_radius

    return positions, radii


def apply_kernel(element_centers, element_size, kernel_positions, kernel_radii):

    # Expand the kernel arrays to allow broadcasting
    nx, ny, nz, _, _ = element_centers.shape
    ns, _ = kernel_positions.shape

    element_size_expanded = np.broadcast_to(element_size, (nx, ny, nz, ns, 1))
    kernel_pos_expanded = np.broadcast_to(kernel_positions, (nx, ny, nz, ns, 3))
    kernel_rad_expanded = np.broadcast_to(kernel_radii, (nx, ny, nz, ns, 1))

    # Broadcast the kernel positions and radii to the grid
    all_positions = element_centers + (element_size_expanded * kernel_pos_expanded)
    all_radii = element_size_expanded * kernel_rad_expanded

    return all_positions, all_radii