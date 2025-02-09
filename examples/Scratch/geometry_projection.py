import numpy as np
import jax.numpy as jnp
import pyvista as pv
from SPI2py.models.projection.grid import create_grid
from SPI2py.models.mechanics.transformations_rigidbody import transform_points
from SPI2py.models.projection.projection import project_component, combine_densities
from SPI2py.models.utilities.visualization import plot_grid, plot_spheres, plot_AABB, plot_stl_file
from SPI2py.models.projection.grid_kernels import create_uniform_kernel
from SPI2py.models.physics.distributed.mesh import generate_mesh, find_active_nodes, find_face_nodes
from SPI2py.models.physics.distributed.assembly import apply_dirichlet_bc, apply_robin_bc, apply_load
from SPI2py.models.physics.distributed.solver import fea_3d_thermal
from SPI2py.models.utilities.visualization import plot_temperature_distribution

# Create grid
el_size = 0.5
el_centers = create_grid(0, 2, 0, 4, 0,  2, element_size=el_size)
nx, ny, nz, _, _ = el_centers.shape
lx, ly, lz = nx * el_size, ny * el_size, nz * el_size

# Read the mesh kernel
kernel_pos, kernel_rad = create_uniform_kernel(1, mode='circumscription')
kernel_pos = kernel_pos.reshape(-1, 3)
kernel_rad = kernel_rad.reshape(-1, 1)

# Read the part model (Numbers for part 1)
# Slice by minimum radius instead of length to maintain kernel symmetry
S_c = 3.0e-2

# Part 1
xyzr_be = np.loadtxt('csvs/Bot_Eye_5k_300s.csv', delimiter=',')
xyzr_be = xyzr_be[xyzr_be[:, 3] >= S_c]
pos_be = xyzr_be[:, :3]
rad_be = xyzr_be[:, 3:4]
pos_be_center = np.mean(pos_be, axis=0, keepdims=True)
pos_be = transform_points(pos_be, pos_be_center, translation=(0.625, 0.625, 0.125), rotation=(0, 0, 0))

# Calculate the pseudo-densities
densities_be, sample_positions_be, sample_radii_be = project_component(el_centers, el_size, pos_be, rad_be, kernel_pos, kernel_rad)
densities_combined = combine_densities(densities_be, min_density=2e-2, penalty_factor=1)

# FEA
density = jnp.ones(nx * ny * nz)

# For simplicity, assume all elements are “solid” (density = 1.0)
nodes_temp, elements_temp = generate_mesh(nx, ny, nz, lx, ly, lz)
n_elem = elements_temp.shape[0]

conv_area = (lx * ly) / ((nx + 1) * (ny + 1))

robin_nodes = find_face_nodes(nodes_temp, jnp.array([0.0, 0.0, 1.0]))
dirichlet_nodes = find_face_nodes(nodes_temp, jnp.array([0.0, 0.0, -1.0]))

# Run the FEA pipeline.
nodes, elements, T = fea_3d_thermal(nx, ny, nz,
                                    lx, ly, lz,
                                    base_k=1.0,
                                    density=densities_combined.flatten(),  # density,
                                    h=10.0,  # Convection coefficient
                                    T_inf=300.0,  # Ambient temperature for convection
                                    fixed_nodes=dirichlet_nodes,
                                    fixed_values=200,
                                    convection_nodes=robin_nodes,
                                    conv_area=conv_area)


T_np = np.array(T)
print("Computed nodal temperatures (sample):", T_np[:10])
nodes_plot = np.array(nodes)
T_plot = np.array(T)


# Plot
plotter = pv.Plotter(shape=(2, 3), window_size=(1500, 500))

# Plot the grid without the kernel
plot_grid(plotter, (0, 0), el_centers, el_size, densities=None)
plot_stl_file(plotter, (0, 0), 'models/Bot_Eye_scaled.stl', translation=(0.625, 0.625, 0.125), rotation=(0, 0, 0))
plot_AABB(plotter, (0, 0), pos_be, rad_be, color='blue')


# Plot the grid with the kernel
plot_AABB(plotter, (1, 1), sample_positions_be, sample_radii_be, color='black', opacity=0.0)
plot_spheres(plotter, (1, 1), sample_positions_be, sample_radii_be, 'blue', opacity=0.5)

plot_spheres(plotter, (0, 1), pos_be, rad_be, 'blue', opacity=0.5)
plot_stl_file(plotter, (0, 1), 'models/Bot_Eye_scaled.stl', translation=(0.625, 0.625, 0.125), rotation=(0, 0, 0), opacity=0.5)

# Plot the grid without the kernel
plot_grid(plotter, (0, 2), el_centers, el_size, densities=densities_combined)

plot_temperature_distribution(plotter,
                              (0, 2),
                              nodes_plot,
                              T_plot,
                              robin_nodes,
                              dirichlet_nodes,
                              (nx + 1, ny + 1, nz + 1))

# plotter.show_axes()
# plotter.link_views()
plotter.show()

