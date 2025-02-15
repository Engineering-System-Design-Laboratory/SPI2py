# import numpy as np
#
# from SPI2py.models.projection.project_interconnects import calculate_combined_densities as ccd1
# from SPI2py.models.projection.project_interconnects_vectorized import calculate_combined_densities as ccd2
#
# # TODO Import plotting...
#
# # Define a fixed-grid mesh of size 4x4x2 with element length 1
# # Represent each element as an inscribing sphere
#
# el_pos = np.array([[[[0., 0., 0.],
#                      [0., 0., 1.]],
#                     [[0., 1., 0.],
#                      [0., 1., 1.]],
#                     [[0., 2., 0.],
#                      [0., 2., 1.]],
#                     [[0., 3., 0.],
#                      [0., 3., 1.]]],
#                    [[[1., 0., 0.],
#                      [1., 0., 1.]],
#                     [[1., 1., 0.],
#                      [1., 1., 1.]],
#                     [[1., 2., 0.],
#                      [1., 2., 1.]],
#                     [[1., 3., 0.],
#                      [1., 3., 1.]]],
#                    [[[2., 0., 0.],
#                      [2., 0., 1.]],
#                     [[2., 1., 0.],
#                      [2., 1., 1.]],
#                     [[2., 2., 0.],
#                      [2., 2., 1.]],
#                     [[2., 3., 0.],
#                      [2., 3., 1.]]],
#                    [[[3., 0., 0.],
#                      [3., 0., 1.]],
#                     [[3., 1., 0.],
#                      [3., 1., 1.]],
#                     [[3., 2., 0.],
#                      [3., 2., 1.]],
#                     [[3., 3., 0.],
#                      [3., 3., 1.]]]])
#
# el_rad = np.array([[[0.5, 0.5],
#                     [0.5, 0.5],
#                     [0.5, 0.5],
#                     [0.5, 0.5]],
#                    [[0.5, 0.5],
#                     [0.5, 0.5],
#                     [0.5, 0.5],
#                     [0.5, 0.5]],
#                    [[0.5, 0.5],
#                     [0.5, 0.5],
#                     [0.5, 0.5],
#                     [0.5, 0.5]],
#                    [[0.5, 0.5],
#                     [0.5, 0.5],
#                     [0.5, 0.5],
#                     [0.5, 0.5]]])
#
# # Define a set of 2 interconnect segments
#
# # Start positions
# x1 = np.array([[0.5, 0.5, 0.5],
#                [3.5, 0.5, 0.5]])
#
# # Stop positions
# x2 = np.array([[3.5, 0.5, 0.5],
#                [3.5, 1.5, 1.5]])
#
# # Radii
# r = np.array([[0.5],
#               [0.5]])
#
