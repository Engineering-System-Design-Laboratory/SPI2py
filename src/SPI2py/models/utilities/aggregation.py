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
    ks_aggregated_max = max_constraint + jnp.log(jnp.sum(jnp.exp(rho * shifted_constraints), axis=axis, keepdims=True)) / rho

    # Remove singleton dimensions if axis was specified
    if axis is not None:
        ks_aggregated_max = jnp.squeeze(ks_aggregated_max, axis=axis)

    return ks_aggregated_max


def kreisselmeier_steinhauser_min(constraints, rho=100, axis=None):
    """
    Computes the Kreisselmeier-Steinhauser (KS) aggregation for the minimum constraint value,
    accounting for operator overflow, using JAX.

    Parameters:
    - constraints: A JAX array containing constraint values.
    - rho: A positive scalar that controls the sharpness of the approximation.
    - axis: The axis along which to apply the KS aggregation.

    Returns:
    - The smooth maximum of the constraint values along the specified axis.
    """

    # Negate the constraints to calculate the minimum
    neg_constraints = -constraints

    # Avoid overflow by subtracting the maximum of the negated constraints
    max_neg_constraint = jnp.max(neg_constraints, axis=axis, keepdims=True)
    shifted_neg_constraints = neg_constraints - max_neg_constraint

    # Compute the KS aggregation
    ks_aggregated_neg = max_neg_constraint + jnp.log(jnp.sum(jnp.exp(rho * shifted_neg_constraints), axis=axis, keepdims=True)) / rho

    # Negate the result to get the smooth minimum
    ks_aggregated_min = -ks_aggregated_neg

    return ks_aggregated_min


