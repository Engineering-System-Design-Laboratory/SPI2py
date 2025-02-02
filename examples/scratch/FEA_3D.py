import numpy as np
import pyvista as pv
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def plot_problem(nx, ny, nz, spacing, densities, boundary_conditions, loads):

    # Plot grid
    x = np.linspace(0, (nx-1) * spacing, nx)
    y = np.linspace(0, (ny-1) * spacing, ny)
    z = np.linspace(0, (nz-1) * spacing, nz)
    grid = pv.RectilinearGrid(x, y, z)
    grid.cell_arrays['Density'] = densities.flatten(order='F')  # Flatten the 3D array to 1D

def shape_function(N, xi, eta, zeta):
    if N == 0:
        return 0.125 * (1 - xi) * (1 - eta) * (1 - zeta)
    elif N == 1:
        return 0.125 * (1 + xi) * (1 - eta) * (1 - zeta)
    elif N == 2:
        return 0.125 * (1 + xi) * (1 + eta) * (1 - zeta)
    elif N == 3:
        return 0.125 * (1 - xi) * (1 + eta) * (1 - zeta)
    elif N == 4:
        return 0.125 * (1 - xi) * (1 - eta) * (1 + zeta)
    elif N == 5:
        return 0.125 * (1 + xi) * (1 - eta) * (1 + zeta)
    elif N == 6:
        return 0.125 * (1 + xi) * (1 + eta) * (1 + zeta)
    elif N == 7:
        return 0.125 * (1 - xi) * (1 + eta) * (1 + zeta)

# Derivatives of shape functions with respect to xi, eta, and zeta
def dshape_function(N, xi, eta, zeta):
    if N == 0:
        return np.array([-0.125 * (1 - eta) * (1 - zeta),
                         -0.125 * (1 - xi) * (1 - zeta),
                         -0.125 * (1 - xi) * (1 - eta)])
    elif N == 1:
        return np.array([0.125 * (1 - eta) * (1 - zeta),
                         -0.125 * (1 + xi) * (1 - zeta),
                         -0.125 * (1 + xi) * (1 - eta)])
    elif N == 2:
        return np.array([0.125 * (1 + eta) * (1 - zeta),
                         0.125 * (1 + xi) * (1 - zeta),
                         -0.125 * (1 + xi) * (1 + eta)])
    elif N == 3:
        return np.array([-0.125 * (1 + eta) * (1 - zeta),
                         0.125 * (1 - xi) * (1 - zeta),
                         -0.125 * (1 - xi) * (1 + eta)])
    elif N == 4:
        return np.array([-0.125 * (1 - eta) * (1 + zeta),
                         -0.125 * (1 - xi) * (1 + zeta),
                         0.125 * (1 - xi) * (1 - eta)])
    elif N == 5:
        return np.array([0.125 * (1 - eta) * (1 + zeta),
                         -0.125 * (1 + xi) * (1 + zeta),
                         0.125 * (1 + xi) * (1 - eta)])
    elif N == 6:
        return np.array([0.125 * (1 + eta) * (1 + zeta),
                         0.125 * (1 + xi) * (1 + zeta),
                         0.125 * (1 + xi) * (1 + eta)])
    elif N == 7:
        return np.array([-0.125 * (1 + eta) * (1 + zeta),
                         0.125 * (1 - xi) * (1 + zeta),
                         0.125 * (1 - xi) * (1 + eta)])


def jacobian_matrix(nodes, xi, eta, zeta):
    J = np.zeros((3, 3))  # 3x3 Jacobian matrix for 3D

    for N in range(8):  # Loop over 8 nodes
        dN_dxi = dshape_function(N, xi, eta, zeta)
        J[:, 0] += dN_dxi[0] * nodes[N, :]
        J[:, 1] += dN_dxi[1] * nodes[N, :]
        J[:, 2] += dN_dxi[2] * nodes[N, :]

    return J



def B_matrix(nodes, xi, eta, zeta):
    J = jacobian_matrix(nodes, xi, eta, zeta)
    J_inv = np.linalg.inv(J)

    B = np.zeros((6, 24))  # 6 strains, 3 displacement components per node for 8 nodes

    for N in range(8):  # Loop over 8 nodes
        dN_dxi = dshape_function(N, xi, eta, zeta)
        dN_dxyz = J_inv @ dN_dxi  # Transform to physical coordinates

        # Populate the B-matrix
        B[0, 3 * N] = dN_dxyz[0]  # B11
        B[1, 3 * N + 1] = dN_dxyz[1]  # B22
        B[2, 3 * N + 2] = dN_dxyz[2]  # B33
        B[3, 3 * N] = dN_dxyz[1]  # B12
        B[3, 3 * N + 1] = dN_dxyz[0]  # B21
        B[4, 3 * N + 1] = dN_dxyz[2]  # B23
        B[4, 3 * N + 2] = dN_dxyz[1]  # B32
        B[5, 3 * N] = dN_dxyz[2]  # B13
        B[5, 3 * N + 2] = dN_dxyz[0]  # B31

    return B


# Define isotropic elasticity matrix D (assuming plane stress for simplicity)
def elasticity_matrix(E, nu):
    C = E / ((1 + nu) * (1 - 2 * nu))
    D = np.array([[C * (1 - nu), C * nu, C * nu, 0, 0, 0],
                  [C * nu, C * (1 - nu), C * nu, 0, 0, 0],
                  [C * nu, C * nu, C * (1 - nu), 0, 0, 0],
                  [0, 0, 0, C * (0.5 - nu), 0, 0],
                  [0, 0, 0, 0, C * (0.5 - nu), 0],
                  [0, 0, 0, 0, 0, C * (0.5 - nu)]])
    return D


# def local_stiffness_matrix(nodes, E, nu):
#     D = elasticity_matrix(E, nu)
#     K = np.zeros((24, 24))  # 8 nodes * 3 DOFs per node = 24 DOFs total
#
#     # Iterate over Gauss points
#     for i in range(2):
#         for j in range(2):
#             for k in range(2):
#                 xi, eta, zeta = gauss_points[i], gauss_points[j], gauss_points[k]
#                 weight = gauss_weights[i] * gauss_weights[j] * gauss_weights[k]
#
#                 B = B_matrix(nodes, xi, eta, zeta)
#                 J = jacobian_matrix(nodes, xi, eta, zeta)
#                 detJ = np.linalg.det(J)
#
#                 K += weight * (B.T @ D @ B) * detJ
#
#     return K

# Reuse previous shape function and B-matrix implementations
# def local_stiffness_matrix_element(nodes, thermal_conductivity):
#     # Similar to what we did earlier for HEX8 element stiffness, but for thermal analysis
#     D = thermal_conductivity * np.eye(3)  # Isotropic material
#     K_local = np.zeros((24, 24))  # 8 nodes * 3 DOFs per node
#
#     # Gauss quadrature points
#     gauss_points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
#     gauss_weights = np.array([1.0, 1.0])
#
#     for i in range(2):
#         for j in range(2):
#             for k in range(2):
#                 xi, eta, zeta = gauss_points[i], gauss_points[j], gauss_points[k]
#                 weight = gauss_weights[i] * gauss_weights[j] * gauss_weights[k]
#
#                 B = B_matrix(nodes, xi, eta, zeta)
#                 J = jacobian_matrix(nodes, xi, eta, zeta)
#                 detJ = np.linalg.det(J)
#
#                 K_local += weight * (B.T @ D @ B) * detJ
#
#     return K_local

def thermal_local_stiffness_matrix(nodes, thermal_conductivity):
    K_local = np.zeros((8, 8))  # 8 nodes with 1 temperature DOF per node

    # Gauss quadrature points
    gauss_points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    gauss_weights = np.array([1.0, 1.0])

    for i in range(2):
        for j in range(2):
            for k in range(2):
                xi, eta, zeta = gauss_points[i], gauss_points[j], gauss_points[k]
                weight = gauss_weights[i] * gauss_weights[j] * gauss_weights[k]

                # Compute Jacobian and its determinant
                J = jacobian_matrix(nodes, xi, eta, zeta)
                detJ = np.linalg.det(J)
                J_inv = np.linalg.inv(J)

                # Gradient of the shape functions in global coordinates
                B = np.zeros((3, 8))  # 3D space gradient for 8 nodes
                for n in range(8):
                    dN_dxi = dshape_function(n, xi, eta, zeta)
                    B[:, n] = J_inv @ dN_dxi  # Transform gradients to global coordinates

                # Local stiffness matrix contribution
                K_local += thermal_conductivity * (B.T @ B) * detJ * weight

    return K_local


def generate_nodal_coordinates(grid_size, element_size):
    nodal_coords = np.zeros((grid_size, grid_size, grid_size, 3))

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                nodal_coords[i, j, k] = [i * element_size, j * element_size, k * element_size]

    return nodal_coords


def get_element_nodes(nodal_coordinates, i, j, k):
    """
    Retrieves the coordinates of the 8 nodes for the element at position (i, j, k).
    """
    element_nodes = np.zeros((8, 3))

    # Get the coordinates of the 8 nodes of the element
    element_nodes[0] = nodal_coordinates[i, j, k]
    element_nodes[1] = nodal_coordinates[i + 1, j, k]
    element_nodes[2] = nodal_coordinates[i + 1, j + 1, k]
    element_nodes[3] = nodal_coordinates[i, j + 1, k]
    element_nodes[4] = nodal_coordinates[i, j, k + 1]
    element_nodes[5] = nodal_coordinates[i + 1, j, k + 1]
    element_nodes[6] = nodal_coordinates[i + 1, j + 1, k + 1]
    element_nodes[7] = nodal_coordinates[i, j + 1, k + 1]

    return element_nodes

# Example element nodal coordinates (cube with unit length)
nodes = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])

J = jacobian_matrix(nodes, 0, 0, 0)


B = B_matrix(nodes, 0, 0, 0)

# Define Gauss points and weights for a 2x2x2 integration (3D)
gauss_points = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
gauss_weights = np.array([1.0, 1.0])

# Example parameters
E = 210e9  # Young's modulus (Pa)
nu = 0.3   # Poisson's ratio


# Generate nodal coordinates for a 10x10x10 grid
grid_size = 10
element_size = 1.0
nodal_coordinates = generate_nodal_coordinates(grid_size, element_size)

# Define material properties
thermal_conductivity = 1.0  # Example value (in W/mK)

# Grid dimensions
element_size = 1.0  # Assuming each element has a size of 1x1x1 for simplicity

# Initialize global stiffness matrix and heat flux vector
num_nodes = grid_size ** 3
global_K = np.zeros((num_nodes, num_nodes))  # Global stiffness matrix
global_F = np.zeros(num_nodes)  # Global heat flux vector (assuming no internal heat sources)

# Loop over each element in the grid to compute the local stiffness matrix and assemble it into the global stiffness matrix
for i in range(grid_size - 1):  # Loop over elements in the x-direction
    for j in range(grid_size - 1):  # Loop over elements in the y-direction
        for k in range(grid_size - 1):  # Loop over elements in the z-direction
            # Get the nodal coordinates for the current element
            element_nodes = get_element_nodes(nodal_coordinates, i, j, k)

            # Compute the local stiffness matrix for this element
            K_local = thermal_local_stiffness_matrix(element_nodes, thermal_conductivity)

            # Here you need to map the local element nodes to global node indices
            # Map the 8 local nodes to their global node indices in the global stiffness matrix
            global_node_indices = [
                i * grid_size ** 2 + j * grid_size + k,
                (i + 1) * grid_size ** 2 + j * grid_size + k,
                (i + 1) * grid_size ** 2 + (j + 1) * grid_size + k,
                i * grid_size ** 2 + (j + 1) * grid_size + k,
                i * grid_size ** 2 + j * grid_size + (k + 1),
                (i + 1) * grid_size ** 2 + j * grid_size + (k + 1),
                (i + 1) * grid_size ** 2 + (j + 1) * grid_size + (k + 1),
                i * grid_size ** 2 + (j + 1) * grid_size + (k + 1)
            ]

            # Assemble the local stiffness matrix into the global stiffness matrix
            for local_i in range(8):
                for local_j in range(8):
                    global_K[global_node_indices[local_i], global_node_indices[local_j]] += K_local[local_i, local_j]

import matplotlib.pyplot as plt

# Assuming `temperature_grid` is the temperature distribution after solving the FEA system.
# Here, we're creating a mock temperature grid for demonstration purposes.
grid_size = 10
temperature_grid = np.linspace(20, 35, grid_size**3).reshape((grid_size, grid_size, grid_size))

# Plot a slice of the temperature grid at the middle along the z-axis (z = 5)
z_slice = temperature_grid[:, :, grid_size // 2]

# Create a 2D plot of the temperature distribution at z = 5
plt.imshow(z_slice, cmap='hot', interpolation='nearest')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution on z = 5 Slice')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# update