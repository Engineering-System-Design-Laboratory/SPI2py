import jax.numpy as np

from .distance_calculations import signed_distances_spheres_spheres



def discrete_collision_detection(x, model, object_pair, object_class_1, object_class_2):
    """
    Returns the signed distances between all pairs of objects in the layout.

    To be consistent with constraint function notation, this function returns negative values
    for objects that are not interfering with each other, and positive values for objects that
    are interfering with each other.

    TODO Preallocate array and vectorize this function
    TODO Write unit tests

    :param x: Design vector (1D array)
    :param model: The SpatialConfiguration object used to query positions at x
    :param object_pair: The list of object pairs to calculate the signed distance between
    :return: An array of signed distances between each object pair
    """
    # Calculate the positions of all spheres in layout given design vector x
    positions_dict = model.calculate_positions(design_vector=x)

    # Calculate the interferences between each sphere of each object pair
    all_signed_distances = []

    for obj1, obj2 in object_pair:

        positions_a = positions_dict[str(obj1)]['positions']
        radii_a = positions_dict[str(obj1)]['radii']

        positions_b = positions_dict[str(obj2)]['positions']
        radii_b = positions_dict[str(obj2)]['radii']


        signed_distances = signed_distances_spheres_spheres(positions_a, radii_a, positions_b, radii_b).flatten()
        all_signed_distances.append(signed_distances)


    all_signed_distances = np.concatenate(all_signed_distances, axis=0)

    return all_signed_distances
