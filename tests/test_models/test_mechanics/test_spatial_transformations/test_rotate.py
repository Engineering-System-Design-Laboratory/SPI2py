# """
# TODO write tests to ensure negative values are same e.g., 90 degrees ccw (-90) = 270 degrees cw (270)
# """
#
# import numpy as np
# from src.SPI2py.computational_model.analysis import rotate
#
#
# def test_rotate_x_90():
#
#     positions = np.array([[2., 2., 0.], [4., 2., 0.], [4., 0., 0.]])
#
#     rotation_x_90 = np.array([np.pi / 2., 0., 0.])
#
#     # Round to avoid 0.9999 != 1, but keep at least one decimal point to avoid njit errors
#     # Njit requires floats not integers!
#     new_positions = np.round(rotate(positions, rotation_x_90), 0)
#
#     # See accompanying PPT file for test case derivations
#     expected = np.array([[2., 2., 0.], [4., 2., 0.], [4., 2., -2.]])
#
#     assert np.array_equal(new_positions, expected)
#
#
# # def test_rotate_y_cc():
# #     new_angle_y_90 = np.array([0., np.pi / 2, 0.])
# #     delta_rotation = new_angle_y_90 - current_angle
# #     new_positions = np.round(rotate(positions, delta_rotation), 0)
# #
# #     expected = np.array([[2., 2., 0.], [2., 2., -2.], [2., 0., -2.]])
# #
# #     assert np.array_equal(new_positions, expected)
# #
# #
# # def test_rotate_z_cc():
# #     new_angle_z_90 = np.array([0., 0., np.pi / 2])
# #     delta_rotation = new_angle_z_90 - current_angle
# #     new_positions = np.round(rotate(positions, delta_rotation), 0)
# #
# #     expected = np.array([[2., 2., 0.], [2., 4., 0.], [4., 4., 0.]])
# #
# #     assert np.array_equal(new_positions, expected)
#
#
# def test_rotation_at_origin_zero():
#
#     # Define a unit cube
#     component_positions = np.array([[0, 0, 0],  # a
#                                     [1, 0, 0],  # b
#                                     [1, 1, 0],  # c
#                                     [0, 1, 0],  # d
#
#                                     [0, 0, 1],  # e
#                                     [1, 0, 1],  # f
#                                     [1, 1, 1],  # g
#                                     [0, 1, 1]])  # h
#
#
#     # Rotate about the origin by 0 degrees
#     rotate_zero = np.array([0, 0, 0])
#
#     # Rotate the cube
#
#     rotated_component_positions = rotate(component_positions, rotate_zero)
#
#     expected_rotated_component_positions_zero = np.array([[0, 0, 0],  # a
#                                                           [1, 0, 0],  # b
#                                                           [1, 1, 0],  # c
#                                                           [0, 1, 0],  # d
#
#                                                           [0, 0, 1],  # e
#                                                           [1, 0, 1],  # f
#                                                           [1, 1, 1],  # g
#                                                           [0, 1, 1]])  # h
#
#     assert np.allclose(rotated_component_positions, expected_rotated_component_positions_zero)
#
#
# def test_rotation_at_origin_z_90():
#
#     # Define a unit cube
#     component_positions = np.array([[0, 0, 0],  # a
#                                     [1, 0, 0],  # b
#                                     [1, 1, 0],  # c
#                                     [0, 1, 0],  # d
#
#                                     [0, 0, 1],  # e
#                                     [1, 0, 1],  # f
#                                     [1, 1, 1],  # g
#                                     [0, 1, 1]])  # h
#
#
#     # Rotate about the z-axis by 90 degrees
#     rotate_z_90 = np.array([0, 0, np.pi/2])
#
#
#     # Rotate the cube
#
#     rotated_component_positions = rotate(component_positions, rotate_z_90)
#
#     expected_rotated_component_positions_z_90 = np.array([[0, 0, 0],  # a
#                                                           [0, 1, 0],  # b
#                                                           [-1, 1, 0],  # c
#                                                           [-1, 0, 0],  # d
#
#                                                           [0, 0, 1],  # e
#                                                           [0, 1, 1],  # f
#                                                           [-1, 1, 1],  # g
#                                                           [-1, 0, 1]])  # h
#
#     assert np.allclose(rotated_component_positions, expected_rotated_component_positions_z_90)
#
# def test_rotation_at_origin_z_180():
#
#     # Define a unit cube
#     component_positions = np.array([[0, 0, 0],  # a
#                                     [1, 0, 0],  # b
#                                     [1, 1, 0],  # c
#                                     [0, 1, 0],  # d
#
#                                     [0, 0, 1],  # e
#                                     [1, 0, 1],  # f
#                                     [1, 1, 1],  # g
#                                     [0, 1, 1]])  # h
#
#     # Rotate about the z-axis by 180 degrees
#     rotate_z_180 = np.array([0, 0, np.pi])
#
#
#     # Rotate the cube
#
#     rotated_component_positions = rotate(component_positions, rotate_z_180)
#
#     expected_rotated_component_positions_z_180 = np.array([[0, 0, 0],  # a
#                                                            [-1, 0, 0],  # b
#                                                            [-1, -1, 0],  # c
#                                                            [0, -1, 0],  # d
#
#                                                            [0, 0, 1],  # e
#                                                            [-1, 0, 1],  # f
#                                                            [-1, -1, 1],  # g
#                                                            [0, -1, 1]])  # h
#
#     assert np.allclose(rotated_component_positions, expected_rotated_component_positions_z_180)
#
# def test_rotation_at_origin_x_90():
#     # Define a unit cube
#     component_positions = np.array([[0, 0, 0],  # a
#                                     [1, 0, 0],  # b
#                                     [1, 1, 0],  # c
#                                     [0, 1, 0],  # d
#
#                                     [0, 0, 1],  # e
#                                     [1, 0, 1],  # f
#                                     [1, 1, 1],  # g
#                                     [0, 1, 1]])  # h
#
#     # Rotate about the x-axis by 90 degrees
#     rotate_x_90 = np.array([np.pi / 2, 0, 0])
#
#     expected_rotated_component_positions_x_90 = np.array([[0, 0, 0],  # a
#                                                           [1, 0, 0],  # b
#                                                           [1, 0, 1],  # c
#                                                           [0, 0, 1],  # d
#
#                                                           [0, -1, 0],  # e
#                                                           [1, -1, 0],  # f
#                                                           [1, -1, 1],  # g
#                                                           [0, -1, 1]])  # h
#
#     # Rotate the cube
#     rotated_component_positions = rotate(component_positions, rotate_x_90)
#
#     assert np.allclose(rotated_component_positions, expected_rotated_component_positions_x_90)
#
# def test_rotation_at_origin_z_90_x_90():
#     # Define a unit cube
#     component_positions = np.array([[0, 0, 0],  # a
#                                     [1, 0, 0],  # b
#                                     [1, 1, 0],  # c
#                                     [0, 1, 0],  # d
#
#                                     [0, 0, 1],  # e
#                                     [1, 0, 1],  # f
#                                     [1, 1, 1],  # g
#                                     [0, 1, 1]])  # h
#
#     rotate_x_90_z_90 = np.array([np.pi / 2, 0, np.pi / 2])
#
#     # Rotate the cube
#     rotated_component_positions = rotate(component_positions, rotate_x_90_z_90)
#
#     expected_rotated_component_positions_z_90_x_90 = np.array([[0, 0, 0],  # a
#                                                                [0, 1, 0],  # b
#                                                                [0, 1, 1],  # c
#                                                                [0, 0, 1],  # d
#
#                                                                [1, 0, 0],  # e
#                                                                [1, 1, 0],  # f
#                                                                [1, 1, 1],  # g
#                                                                [1, 0, 1]])  # h
#
#     assert np.allclose(rotated_component_positions, expected_rotated_component_positions_z_90_x_90)