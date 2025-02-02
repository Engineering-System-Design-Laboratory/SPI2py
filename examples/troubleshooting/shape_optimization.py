import numpy as np
from scipy.optimize import minimize
from volume import total_overlap_volume
from spherical_decomposition import pseudo_mdbd_rect_prism

def volume(x):
    l, w, h = x
    return l * w * h

def volume_mdbd(x):
    l, w, h = x
    x_min, x_max = 0, l
    y_min, y_max = 0, w
    z_min, z_max = 0, h
    pos, rad = pseudo_mdbd_rect_prism(x_min, x_max, y_min, y_max, z_min, z_max,
                                      num_spheres=2500,
                                      min_radius=1e-6,
                                      meshgrid_increment=200)
    v = np.sum(4/3 * np.pi * rad**3)
    return v


# ov = total_overlap_volume(sphere_points, sphere_radii)
# print(f"Overlap volume: {ov}")


# bounds = [(0.2, 3), (0.2, 3), (0.2, 3)]
# initial_guess = [1, 1, 1]
# res1 = minimize(volume, initial_guess, bounds=bounds)
#
# print(res1)
#
# res2 = minimize(volume_mdbd, initial_guess, bounds=bounds)
#
# print(res2)
#
# # v_analytical = volume(initial_guess)
# # v_mdbd = volume_mdbd(initial_guess)
# #
# # print(f"Analytical volume: {v_analytical}")
# # print(f"MDBD volume: {v_mdbd}")