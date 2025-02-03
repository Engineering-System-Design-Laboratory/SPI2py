import jax.numpy as jnp
from typing import Tuple, Union


def assert_jax_array_shape_dtype(
    arr: jnp.ndarray,
    *,
    expected_shape: Tuple[Union[int, None], ...],
    expected_dtype: Union[jnp.dtype, str, None] = None,
    name: str = "array",
) -> None:
    """
    Validates the shape and dtype of a JAX array.

    Parameters:
    ----------
    arr : jnp.ndarray
        The JAX array to validate.
    expected_shape : Tuple[Union[int, None], ...]
        Expected shape pattern, where each dimension can be:
          - an integer (exact size),
          - None or -1 (free dimension).
    expected_dtype : Union[np.dtype, str, None], optional
        Expected dtype of the array. If None, dtype is not checked.
    name : str, optional
        A name for the array, used in error messages.
    """
    # 1. Check if input is a JAX array
    if not isinstance(arr, jnp.ndarray):
        raise ValueError(f"{name} must be a JAX array, got {type(arr)} instead.")

    # 2. Check shape
    actual_shape = arr.shape
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"{name} has {len(actual_shape)} dimensions, expected {len(expected_shape)} "
            f"(shape={actual_shape}, expected={expected_shape})."
        )

    for i, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected_shape)):
        if expected_dim in (None, -1):
            continue  # Free dimension, no check needed
        if actual_dim != expected_dim:
            raise ValueError(
                f"{name} dimension {i} mismatch: got {actual_dim}, expected {expected_dim}. "
                f"(shape={actual_shape}, expected={expected_shape})"
            )

    # 3. Check dtype
    if expected_dtype is not None:
        if isinstance(expected_dtype, str):
            expected_dtype = jnp.dtype(expected_dtype)
        if arr.dtype != expected_dtype:
            raise ValueError(
                f"{name} has dtype {arr.dtype}, expected {expected_dtype}."
            )
