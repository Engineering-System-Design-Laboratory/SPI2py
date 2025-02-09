import numpy as np
import pyvista as pv
import jax.numpy as jnp
from SPI2py.models.physics.distributed.mesh import generate_mesh, find_active_nodes, find_face_nodes
from SPI2py.models.physics.distributed.assembly import apply_dirichlet_bc, apply_robin_bc, apply_load
from SPI2py.models.physics.distributed.solver import fea_3d_thermal
from SPI2py.models.utilities.visualization import plot_temperature_distribution


# Domain and discretization parameters:
nx, ny, nz = 5, 5, 5
lx, ly, lz = 2.0, 2.0, 2.0


# For simplicity, assume all elements are “solid” (density = 1.0)
nodes_temp, elements_temp = generate_mesh(nx, ny, nz, lx, ly, lz)
n_elem = elements_temp.shape[0]
density = jnp.ones(n_elem)



# Assume each convection node represents an equal share of the top surface area.
conv_area = (lx * ly) / ((nx + 1) * (ny + 1))


# Define Dirichlet BCs: for example, the bottom face (z=0) is fixed at 300.
robin_nodes = find_face_nodes(nodes_temp, jnp.array([0.0, 0.0, 1.0]))
dirichlet_nodes = find_face_nodes(nodes_temp, jnp.array([0.0, 0.0, -1.0]))

# Run the FEA pipeline.
nodes, elements, T = fea_3d_thermal(nx, ny, nz,
                                    lx, ly, lz,
                                    base_k=1.0,
                                    density=density,
                                    h=10.0,  # Convection coefficient
                                    T_inf=300.0,  # Ambient temperature for convection
                                    fixed_nodes=dirichlet_nodes,
                                    fixed_values=200,
                                    convection_nodes=robin_nodes,
                                    conv_area=conv_area)

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
plot_temperature_distribution(plotter, (0, 0), nodes_plot, T_plot, robin_nodes, dirichlet_nodes, dims=dims)