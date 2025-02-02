from numpy import np





def create_cylinders(points, radius):
    x1 = points[:-1]  # Start positions (-1, 3)
    x2 = points[1:]   # Stop positions (-1, 3)
    r  = np.full((x1.shape[0], 1), radius)
    return x1, x2, r


