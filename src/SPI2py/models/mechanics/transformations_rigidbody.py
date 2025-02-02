"""

"""

import jax.numpy as jnp


def transform_points(positions, reference_point, translation, rotation):
    """
    Transforms the given positions by applying rotation and translation with respect to a reference point.

    Parameters:
    - positions: numpy array of shape (N, 3), the points to be transformed.
    - reference_point: numpy array of shape (3,), the reference point for rotation.
    - translation: numpy array of shape (3,), the translation vector.
    - rotation: numpy array of shape (3,), the rotation angles (alpha, beta, gamma) in radians.

    Returns:
    - transformed_positions: numpy array of shape (N, 3), the transformed points.
    """

    # Ensure inputs are proper shapes
    translation = jnp.asarray(translation).reshape(3)
    rotation = jnp.asarray(rotation).reshape(3)
    reference_point = jnp.asarray(reference_point).reshape(3)
    positions = jnp.asarray(positions)

    # Assemble the transformation matrix
    t = jnp.eye(4, dtype=jnp.float64)

    # Unpack the rotation angles (Euler angles)
    alpha, beta, gamma = rotation  # rotation around x, y, z axes respectively

    # Calculate rotation matrix components
    ca, cb, cg = jnp.cos(alpha), jnp.cos(beta), jnp.cos(gamma)
    sa, sb, sg = jnp.sin(alpha), jnp.sin(beta), jnp.sin(gamma)

    # Rotation matrix
    r = jnp.array([
        [cb * cg, sa * sb * cg - ca * sg, ca * sb * cg + sa * sg],
        [cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - sa * cg],
        [-sb,     sa * cb,                ca * cb]
    ])

    # t[:3, :3] = r
    t = t.at[:3, :3].set(r)  # JAX update syntax

    # Insert the translation vector
    # t[:3, 3] = translation
    t = t.at[:3, [3]].set(translation)  # JAX update syntax

    # Apply the transformation matrix
    # Shift positions by the reference point
    positions_shifted = positions - reference_point

    # Convert to homogeneous coordinates by adding a column of ones
    ones = jnp.ones((positions_shifted.shape[0], 1))
    positions_homogeneous = jnp.hstack([positions_shifted, ones])  # Shape: (N, 4)

    # Apply the transformation
    transformed_positions_homogeneous = positions_homogeneous @ t.T  # Shape: (N, 4)

    # Convert back from homogeneous coordinates
    transformed_positions_shifted = transformed_positions_homogeneous[:, :3]

    # Shift back by adding the reference point
    transformed_positions = transformed_positions_shifted + reference_point

    return transformed_positions



