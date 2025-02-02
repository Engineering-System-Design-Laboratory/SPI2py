"""

"""

import jax.numpy as jnp

def assemble_transformation_matrix(translation, rotation):

    # Ensure translation and rotation are proper shapes
    if translation.shape != (3, 1):
        translation = translation.reshape((3, 1))

    if rotation.shape != (3):
        rotation = rotation.reshape((3))

    # Initialize the transformation matrix
    t = jnp.eye(4, dtype=jnp.float64)

    # Insert the translation vector
    # t = t.at[:3, 3].set(translation.flatten())  # JAX update syntax
    t = t.at[:3, [3]].set(translation)  # JAX update syntax

    # Unpack the rotation angles (Euler)
    a, b, g = rotation  # alpha, beta, gamma

    # Calculate rotation matrix components
    ca, cb, cg = jnp.cos(a), jnp.cos(b), jnp.cos(g)
    sa, sb, sg = jnp.sin(a), jnp.sin(b), jnp.sin(g)

    # Calculate rotation matrix (R = R_z(gamma) @ R_y(beta) @ R_x(alpha))
    r = jnp.array([[cb * cg, sa * sb * cg - ca * sg, ca * sb * cg + sa * sg],
                   [cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - sa * cg],
                   [-sb, sa * cb, ca * cb]])

    # Insert the rotation matrix
    t = t.at[:3, :3].set(r)

    return t


def apply_transformation_matrix(reference_point, positions, transformation_matrix):
    """
    Assume transposed...?
    """

    # Center the object about its reference position
    positions_shifted = positions - reference_point

    # Pad the positions with ones
    ones = jnp.ones((1, positions_shifted.shape[1]))
    positions_shifted_padded = jnp.vstack((positions_shifted, ones))

    # Apply the transformation
    transformed_positions_shifted_padded = transformation_matrix @ positions_shifted_padded

    # Remove the padding
    transformed_positions_shifted = transformed_positions_shifted_padded[:3, :]

    # Shift the object back to its original position
    transformed_positions = transformed_positions_shifted + reference_point

    return transformed_positions






