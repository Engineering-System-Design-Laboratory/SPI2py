import numpy as np
import pyvista as pv
from time import time_ns
from mdbd import convert_stl_to_mdbd
from volume import total_overlap_volume
from visualization import plot_spheres

# Define the bounds of the rect prism
x_min, x_max = -0.5, 0.5
y_min, y_max = -0.5, 0.5
z_min, z_max = -0.5, 0.5

# Generate the MDBD
start = time_ns()
sphere_points, sphere_radii = convert_stl_to_mdbd("models/", "CogDrivenGear.stl",
                                                  n_spheres=5000,
                                                  n_steps=300,
                                                  scale=0.1)
stop = time_ns()


# Evaluate the properties of the MDBD
runtime = (stop-start)/1e9
print(f"Runtime: {runtime} s")

true_volume = (x_max-x_min)*(y_max-y_min)*(z_max-z_min)
approx_volume = np.sum(4/3 * np.pi * sphere_radii**3)
print(f"True volume: {true_volume}")
print(f"Approx volume: {approx_volume}")

# Should be zero
overlap_volume = total_overlap_volume(sphere_points, sphere_radii)
print(f"Overlap volume: {overlap_volume}")

# # Plot the spheres
# plotter = pv.Plotter()
# plot_spheres(plotter, (0, 0), sphere_points, sphere_radii)
# plotter.show()

# # Save the sphere centers and radii to a CSV file
xyzr = np.hstack([sphere_points, sphere_radii])
np.savetxt("csvs/CogDrivenGear_5k_300s.csv", xyzr, delimiter=",")
