import jax.numpy as jnp
from chex import assert_shape, assert_type


def kreisselmeier_steinhauser_max(constraints, rho=100, axis=None):
    """
    Computes the Kreisselmeier-Steinhauser (KS) aggregation for the maximum constraint value,
    accounting for operator overflow, using JAX.

    Parameters:
    - constraints: A JAX array containing constraint values.
    - rho: A positive scalar that controls the sharpness of the approximation.
    - axis: The axis along which to apply the KS aggregation.

    Returns:
    - The smooth maximum of the constraint values along the specified axis.
    """
    # Avoid overflow by subtracting the maximum value along the axis
    max_constraint = jnp.max(constraints, axis=axis, keepdims=True)
    shifted_constraints = constraints - max_constraint

    # Compute the KS aggregation
    ks_aggregated_max = max_constraint + jnp.log(jnp.sum(jnp.exp(rho * shifted_constraints), axis=axis)) / rho

    # Remove singleton dimensions if axis was specified
    if axis is not None:
        ks_aggregated_max = jnp.squeeze(ks_aggregated_max, axis=axis)

    return ks_aggregated_max


def apply_rho_min(densities, rho_min=1e-3, rho=100):
    """
    Combines densities using the KS max function and applies a minimum density
    for numerical stability.

    Parameters:
    - densities: A JAX array of pseudo-densities (shape: (..., n_densities)).
    - rho_min: Minimum allowable density.
    - rho: Parameter controlling the sharpness of the KS aggregation.

    Returns:
    - Combined densities with minimum density applied.
    """
    # Apply KS max across the last axis
    combined_density = kreisselmeier_steinhauser_max(densities, rho=rho, axis=-1)

    # Apply rho_min where the original densities were all zero
    all_zero_mask = jnp.all(densities == 0, axis=-1)
    combined_density = jnp.where(all_zero_mask, rho_min, combined_density)

    return combined_density

