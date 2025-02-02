import jax.numpy as jnp


def translate_linear_spline(start_point, control_points, end_point):

    # Combine all control points into a single tensor
    node_positions = jnp.vstack((start_point, control_points, end_point))

    # Obtain the start and stop points for each line segment
    start_positions = node_positions[:-1]
    stop_positions = node_positions[1:]

    # start_positions = node_positions.view(-1, 6)[:, [0, 1, 2]]
    # stop_positions = node_positions.view(-1, 6)[:, [3, 4, 5]]

    # Calculate the translations for each line segment
    translations = jnp.empty((0, 3), dtype=jnp.float64)
    for start_position, stop_position in zip(start_positions, stop_positions):
        segment_positions = compute_line_segment_intermediate_positions(start_position, stop_position, num_spheres_per_segment)
        translations = jnp.vstack((translations, segment_positions))

    # Apply the translations
    translated_positions = positions + translations

    return translated_positions

def compute_line_segment_intermediate_positions(start_position, stop_position, num_spheres_per_segment):

    x_start = start_position[0]
    y_start = start_position[1]
    z_start = start_position[2]

    x_stop = stop_position[0]
    y_stop = stop_position[1]
    z_stop = stop_position[2]

    x_linspace = jnp.linspace(x_start, x_stop, num_spheres_per_segment)
    y_linspace = jnp.linspace(y_start, y_stop, num_spheres_per_segment)
    z_linspace = jnp.linspace(z_start, z_stop, num_spheres_per_segment)

    positions = jnp.vstack((x_linspace, y_linspace, z_linspace)).T

    return positions








