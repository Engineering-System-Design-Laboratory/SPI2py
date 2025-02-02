import numpy as np


def create_grid(x_min, x_max, y_min, y_max, z_min, z_max, element_size=1.0):
    # Shift the range by half the element size to ensure the corner is at (xmin, ymin, zmin)
    x = np.arange(x_min + element_size / 2, x_max, element_size)
    y = np.arange(y_min + element_size / 2, y_max, element_size)
    z = np.arange(z_min + element_size / 2, z_max, element_size)

    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')

    # Stack the coordinates to form a position array
    pos = np.stack((xv, yv, zv), axis=-1)

    # Expand dimensions to include an extra axis as in your original function
    pos = np.expand_dims(pos, axis=3)

    return pos