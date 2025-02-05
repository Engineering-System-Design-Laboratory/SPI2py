# import numpy as np
# from scipy.spatial.distance import cdist
# from SPI2py.models.group_spatial.component_kinematic.distance_calculations import distances_points_points
#
# def test_pairwise_distance():
#     """
#     Calculates the pairwise distance between two sets of points.
#
#     We know that Scipy cdist does this correctly so check with it.
#     """
#     a = np.random.rand(20, 3)
#     b = np.random.rand(17, 3)
#
#     c = distances_points_points(a, b)
#
#     cd = cdist(a, b).reshape(-1)
#
#     assert all(np.isclose(c, cd))
