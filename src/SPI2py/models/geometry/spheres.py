"""Pseudo MDBD
"""

import numpy as np
import pyvista as pv
import vtk


def compute_distances_to_faces(points, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Calculate the distance from each point to the nearest face of the rectangular prism.
    """
    x_distances = np.minimum(points[:, 0] - x_min, x_max - points[:, 0])
    y_distances = np.minimum(points[:, 1] - y_min, y_max - points[:, 1])
    z_distances = np.minimum(points[:, 2] - z_min, z_max - points[:, 2])
    return np.minimum(np.minimum(x_distances, y_distances), z_distances)


def compute_signed_distance(mesh, points, invert=True):

    # Convert PyVista mesh to VTK polydata
    mesh_vtk = mesh

    # Create the vtkImplicitPolyDataDistance object
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(mesh_vtk)

    # Calculate the signed distance for each point
    signed_distances = np.array([implicit_distance.EvaluateFunction(point) for point in points])

    # Invert the distances if needed
    if invert:
        signed_distances *= -1

    return signed_distances


def recurse_mdbd(n_spheres, distances_filtered_sorted, points_filtered_sorted):

    # Preallocate arrays for sphere centers and radii
    sphere_points = np.zeros((n_spheres, 3))
    sphere_radii = np.zeros((n_spheres, 1))

    # Iterate to pack spheres until reaching the limit or the smallest sphere is smaller than min_radius
    for i in range(n_spheres):

        if distances_filtered_sorted.size == 0:
            break

        # Choose the point with the maximum distance from any surface or existing sphere
        sphere_center = points_filtered_sorted[0]
        sphere_radius = distances_filtered_sorted[0]

        # Update lists of points and distances
        sphere_points[i] = sphere_center
        sphere_radii[i] = sphere_radius

        # Update distances considering the newly added sphere
        point_distances_to_new_sphere = np.linalg.norm(points_filtered_sorted - sphere_center, axis=1)
        within_new_sphere = point_distances_to_new_sphere < sphere_radius + distances_filtered_sorted
        points_filtered_sorted = points_filtered_sorted[~within_new_sphere]
        distances_filtered_sorted = distances_filtered_sorted[~within_new_sphere]

    # Trim the arrays to remove unused entries
    sphere_points = sphere_points[:i]
    sphere_radii = sphere_radii[:i]

    return i, sphere_points, sphere_radii


def convert_primitive_to_mdbd(x_min, x_max, y_min, y_max, z_min, z_max,
                              n_spheres=1000, min_radius=1e-3,
                              meshgrid_increment=30):

    # Create a 3D meshgrid within the specified bounds
    x = np.linspace(x_min, x_max, meshgrid_increment)
    y = np.linspace(y_min, y_max, meshgrid_increment)
    z = np.linspace(z_min, z_max, meshgrid_increment)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    all_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Calculate the distances from each point to the nearest face of the prism
    distances_to_faces = compute_distances_to_faces(all_points, x_min, x_max, y_min, y_max, z_min, z_max)

    # Sort points by their distance to the surface (descending order)
    sorted_indices = np.argsort(distances_to_faces)[::-1]
    points_filtered_sorted = all_points[sorted_indices]
    distances_filtered_sorted = distances_to_faces[sorted_indices]

    # Remove points with distances smaller than the minimum radius
    points_filtered_sorted = points_filtered_sorted[distances_filtered_sorted >= min_radius]
    distances_filtered_sorted = distances_filtered_sorted[distances_filtered_sorted >= min_radius]

    i, sphere_points, sphere_radii = recurse_mdbd(n_spheres, distances_filtered_sorted, points_filtered_sorted)

    return sphere_points, sphere_radii


def convert_stl_to_mdbd(directory,
                        filename,
                        n_spheres=1000,
                        n_steps=25,
                        scale=1):

    # Read the mesh
    mesh = pv.read(directory+filename)

    # Create a meshgrid of points
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x = np.linspace(x_min, x_max, n_steps)
    y = np.linspace(y_min, y_max, n_steps)
    z = np.linspace(z_min, z_max, n_steps)
    all_points = np.array(np.meshgrid(x, y, z)).reshape(3, -1).T

    # Calculate inverted signed distances for all points
    signed_distances = compute_signed_distance(mesh, all_points, invert=True)

    # Remove points outside the mesh
    mask_interior = signed_distances > 0
    points_interior = all_points[mask_interior]
    distances_interior = signed_distances[mask_interior]

    # Sort points by their distance from the surface, descending
    sorted_indices = np.argsort(distances_interior)[::-1]
    points_filtered_sorted = points_interior[sorted_indices]
    distances_filtered_sorted = distances_interior[sorted_indices]

    # Scale the distances
    points_filtered_sorted *= scale
    distances_filtered_sorted *= scale

    i, sphere_points, sphere_radii = recurse_mdbd(n_spheres, distances_filtered_sorted, points_filtered_sorted)

    return sphere_points, sphere_radii


def get_aabb_bounds(centers, radii):
    """
    Get the AABB bounds for the object
    """

    # Flatten the arrays
    centers = centers.reshape(-1, 3)
    radii = radii.reshape(-1, 1)

    x_min, y_min, z_min = np.min(centers - radii, axis=0)
    x_max, y_max, z_max = np.max(centers + radii, axis=0)

    return x_min, x_max, y_min, y_max, z_min, z_max


def get_aabb_indices(el_centers, el_size, obj_centers, obj_radii):
    """
    Get the indices of the AABB bounds in the grid
    """

    aabb_bounds = get_aabb_bounds(obj_centers, obj_radii)

    obj_x_min, obj_x_max, obj_y_min, obj_y_max, obj_z_min, obj_z_max = aabb_bounds

    element_half_size = el_size / 2

    el_x_min = el_centers[:, :, :, :, 0].squeeze() - element_half_size
    el_x_max = el_centers[:, :, :, :, 0].squeeze() + element_half_size
    el_y_min = el_centers[:, :, :, :, 1].squeeze() - element_half_size
    el_y_max = el_centers[:, :, :, :, 1].squeeze() + element_half_size
    el_z_min = el_centers[:, :, :, :, 2].squeeze() - element_half_size
    el_z_max = el_centers[:, :, :, :, 2].squeeze() + element_half_size

    # Find the overlapping elements along each dimension
    x_overlap = (obj_x_min <= el_x_max) & (obj_x_max >= el_x_min)
    y_overlap = (obj_y_min <= el_y_max) & (obj_y_max >= el_y_min)
    z_overlap = (obj_z_min <= el_z_max) & (obj_z_max >= el_z_min)

    # Combine overlaps to get the AABB
    overlap = x_overlap & y_overlap & z_overlap

    # Get indices of overlapping elements
    indices = np.argwhere(overlap)

    i1 = np.min(indices[:, 0])
    i2 = np.max(indices[:, 0])
    j1 = np.min(indices[:, 1])
    j2 = np.max(indices[:, 1])
    k1 = np.min(indices[:, 2])
    k2 = np.max(indices[:, 2])

    return i1, i2, j1, j2, k1, k2
