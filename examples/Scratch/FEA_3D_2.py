import numpy as np
import pyvista as pv
import jax.numpy as jnp
from SPI2py.models.physics.FEA.mesh import generate_mesh
from SPI2py.models.physics.FEA.solver import fea_3d_thermal
from SPI2py.models.utilities.visualization import plot_temperature_distribution



# Domain and discretization parameters:
# nx, ny, nz = 10, 10, 10
# lx, ly, lz = 1.0, 1.0, 1.0
nx, ny, nz = 5, 5, 5
lx, ly, lz = 2.0, 2.0, 2.0
base_k = 1.0
penal = 3.0
# For simplicity, assume all elements are “solid” (density = 1.0)
nodes_temp, elements_temp = generate_mesh(nx, ny, nz, lx, ly, lz)
n_elem = elements_temp.shape[0]
density = jnp.ones(n_elem)

# Define convection: let the top face (z = lz) have convection.
tol = 1e-6
nodes_np = np.array(nodes_temp)
convection_nodes = np.where(np.abs(nodes_np[:, 2] - lz) < tol)[0]
# Assume each convection node represents an equal share of the top surface area.
conv_area = (lx * ly) / ((nx + 1) * (ny + 1))
h = 10.0
T_inf = 300.0

# Define Dirichlet BCs: for example, the bottom face (z=0) is fixed at 300.
fixed_nodes = np.where(np.abs(nodes_np[:, 2] - 0.0) < tol)[0]
fixed_values = np.full(fixed_nodes.shape, 300.0)

# Run the FEA pipeline.
nodes, elements, T = fea_3d_thermal(
    nx, ny, nz, lx, ly, lz,
    base_k, penal, density, h, T_inf,
    fixed_nodes, fixed_values, convection_nodes, conv_area
)

# Convert the resulting temperature field to a NumPy array for further processing or plotting.
T_np = np.array(T)
print("Computed nodal temperatures (sample):", T_np[:10])

# Convert the solution to NumPy arrays for plotting.
nodes_plot = np.array(nodes)
T_plot = np.array(T)

# Plot the FEA results.
dims = (nx + 1, ny + 1, nz + 1)
# Plot
plotter = pv.Plotter(shape=(1, 1), window_size=(1500, 500))
plot_temperature_distribution(plotter, (0, 0), nodes_plot, T_plot, convection_nodes, fixed_nodes, dims=dims)