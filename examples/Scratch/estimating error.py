import jax.numpy as jnp
def estimate_partial_derivative_memory(n_points, nx, ny, nz):
    pd_size = n_points*3*nx*ny*nz
    pd = jnp.ones((pd_size, 1), dtype=jnp.float64)

    print(f"Jacobian size: {pd_size}")

    memory_usage_bytes = pd.nbytes
    memory_usage_mb = memory_usage_bytes / (1024 ** 2)
    print(f"Memory usage: {memory_usage_mb:.2f} MB")

def estimate_projection_error(prob, radii, variable, volume, default_set_val, steps, step_size):

    sphere_radii = prob.get_val(radii)
    sphere_volume = 4/3 * jnp.pi * sphere_radii**3
    true_volume = float(jnp.sum(sphere_volume))

    volumes = []

    prob.set_val(variable, default_set_val)
    prob.run_model()
    volumes.append(float(prob.get_val(volume)))

    for i in range(steps):
        default_set_val[0] += step_size
        default_set_val[1] += step_size
        default_set_val[2] += step_size
        prob.set_val(variable, default_set_val)
        prob.run_model()
        volumes.append(float(prob.get_val(volume)))


    volumes = jnp.array(volumes)
    max_relative_error = round(100 * jnp.max(jnp.abs(volumes - volumes[0]) /volumes[0]), 2)
    max_true_error = round(100 * jnp.max(jnp.abs(volumes - true_volume) / true_volume), 2)

    print('Volumes:', volumes)
    print(f'Max error wrt mesh: {max_relative_error} %')
    print(f'Max error wrt mdbd volume: {max_true_error} %')