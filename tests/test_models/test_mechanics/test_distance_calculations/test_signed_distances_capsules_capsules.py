# import numpy as np
# from SPI2py.models.group_spatial.component_kinematic.distance_calculations import signed_distances_capsules_capsules
#
#
# def test_capsule_capsule_tangent_1():
#
#     a = np.array([[0., 0., 0.]])
#     b = np.array([[1., 0., 0.]])
#     ab_radii = np.array([0.5])
#
#     c = np.array([[0., 0., 1.]])
#     d = np.array([[1., 0., 1.]])
#     cd_radii = np.array([0.5])
#
#     distances = signed_distances_capsules_capsules(a, b, ab_radii, c, d, cd_radii)
#     expected_distances = np.array([0.0])
#
#     assert np.allclose(distances, expected_distances)
#
# def test_capsule_capsule_nonoverlap_1():
#
#     a = np.array([[0., 0., 0.]])
#     b = np.array([[1., 0., 0.]])
#     ab_radii = np.array([0.5])
#
#     c = np.array([[0., 0., 2.]])
#     d = np.array([[1., 0., 2.]])
#     cd_radii = np.array([0.5])
#
#     distances = signed_distances_capsules_capsules(a, b, ab_radii, c, d, cd_radii)
#     expected_distances = np.array([-1.0])
#
#     assert np.allclose(distances, expected_distances)
#
#
#
# def test_capsule_capsule_fully_overlapping():
#
#     a = np.array([[0., 0., 0.]])
#     b = np.array([[1., 0., 0.]])
#     ab_radii = np.array([0.5])
#
#     c = np.array([[0., 0., 0.]])
#     d = np.array([[1., 0., 0.]])
#     cd_radii = np.array([0.5])
#
#     distances = signed_distances_capsules_capsules(a, b, ab_radii, c, d, cd_radii)
#     expected_distances = np.array([1.0])
#
#     assert np.allclose(distances, expected_distances)
#
#
# def test_capsules_capsules_nonoverlap_1():
#
#     a = np.array([[0., 0., 0.], [0., 0., 0.]])
#     b = np.array([[1., 0., 0.], [1., 0., 0.]])
#     ab_radii = np.array([0.5, 0.5])
#
#     c = np.array([[0., 0., 2.], [0., 0., 2.]])
#     d = np.array([[1., 0., 2.], [1., 0., 2.]])
#     cd_radii = np.array([0.5, 0.5])
#
#     distances = signed_distances_capsules_capsules(a, b, ab_radii, c, d, cd_radii)
#     expected_distances = np.array([-1.0, -1.0, -1.0, -1.0])
#
#     assert np.allclose(distances, expected_distances)
#
#
# # def test_parallel_horizontal_within_range():
# #
# #     a = np.array([0., 0., 0.])
# #     b = np.array([1., 0., 0.])
# #     c = np.array([0., 0., 1.])
# #     d = np.array([1., 0., 1.])
# #
# #     dist, _ = minimum_distance_capsule_capsule(a, b, c, d)
# #
# #     assert np.isclose(dist, 1.0)
# #
# #
# # def test_parallel_horizontal_out_of_range():
# #
# #     a = np.array([0., 0., 0.])
# #     b = np.array([1., 0., 0.])
# #     c = np.array([2., 0., 1.])
# #     d = np.array([3., 0., 1.])
# #
# #     dist, _ = minimum_distance_capsule_capsule(a, b, c, d)
# #
# #     expected_dist = np.linalg.norm(c-b)
# #
# #     assert np.isclose(dist, expected_dist)
# #
# #
# # def test_parallel_horizontal_along_same_axis():
# #
# #     a = np.array([0., 0., 0.])
# #     b = np.array([1., 0., 0.])
# #     c = np.array([2., 0., 0.])
# #     d = np.array([3., 0., 0.])
# #
# #     dist, _ = minimum_distance_capsule_capsule(a, b, c, d)
# #
# #     assert np.isclose(dist, 1.0)
# #
# #
# # def test_parallel_vertical_within_range():
# #
# #     a = np.array([0., 0., 0.])
# #     b = np.array([0., 1., 0.])
# #     c = np.array([0., 0., 1.])
# #     d = np.array([0., 1., 1.])
# #
# #     dist, _ = minimum_distance_capsule_capsule(a, b, c, d)
# #
# #     assert np.isclose(dist, 1.0)
# #
# #
# # def test_parallel_vertical_out_of_range():
# #
# #     a = np.array([0., 0., 0.])
# #     b = np.array([0., 1., 0.])
# #     c = np.array([1., 2., 0.])
# #     d = np.array([1., 3., 0.])
# #
# #     dist, _ = minimum_distance_capsule_capsule(a, b, c, d)
# #
# #     expected_dist = np.linalg.norm(c - b)
# #
# #     assert np.isclose(dist, expected_dist)
# #
# #
# # def test_skew():
# #
# #     a = np.array([0., 0., 0.])
# #     b = np.array([1., 0., 0.])
# #     c = np.array([0., 0., 2.])
# #     d = np.array([1., 0., 1.])
# #
# #     dist, _ = minimum_distance_capsule_capsule(a, b, c, d)
# #
# #     assert np.isclose(dist, 1.0)