import numpy as np
import matplotlib.pyplot as plt



element_length = 1.0
kernel_scale_factor = 1.0

true_volume = element_length**3


mdbd_volume = 0
mdbd_volumes = []
for i, radius in enumerate(mdbd_1000_kernel_radii):
    mdbd_volume += 4/3 * np.pi * radius**3
    mdbd_volumes.append(mdbd_volume)

mdbd_max_error = mdbd_1000_kernel_radii[1:]


# TODO Verify
def largest_sphere_between_8_spheres(small_sphere_radius):
    # Calculate the radius of the largest sphere
    R = small_sphere_radius * (np.sqrt(3) - 1)

    # Calculate the volume of the largest sphere
    volume = (4 / 3) * np.pi * R ** 3

    return R, volume

#

uniform_volumes = []
n_uniform_spheres = []
largest_error = []
for i in range(1, 11):
    positions, radii = inscribe_spheres_in_cube(i)
    volume = np.sum(4/3 * np.pi * radii**3)
    uniform_volumes.append(volume)
    n_uniform_spheres.append(i**3)
    R, largest_volume = largest_sphere_between_8_spheres(0.5 / i)
    largest_error.append(largest_volume)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the volumes
ax1.plot(mdbd_volumes, label='MDBD')
ax1.plot(n_uniform_spheres, uniform_volumes, label='Uniform')
ax1.axhline(true_volume, color='black', linestyle='--', label='True')
ax1.set_xlim(0, 1000)
ax1.set_ylim(0, 1.1)
ax1.set_xlabel('Number of kernels')
ax1.set_ylabel('Volume')
ax1.legend()

# Plot the errors
ax2.plot(mdbd_max_error, label='MDBD max error')
ax2.plot(n_uniform_spheres, largest_error, label='Uniform max error')
ax2.set_xlim(0, 1000)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('Number of kernels')
ax2.set_ylabel('Error')
ax2.legend()

# Show the figure
plt.tight_layout()
plt.show()

# Note the above 1 corresponds to errors in creating MDBD, should fix

# Save the figure
fig.savefig('figures/mesh_kernel_properties.png')