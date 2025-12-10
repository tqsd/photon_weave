"""
Lightweight PRNG helpers to make key threading explicit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def borrow_key(
    key: Optional[jnp.ndarray],
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Return a split key pair. Requires an explicit key.

    Parameters
    ----------
    key : jnp.ndarray
        PRNG key to split.

    Returns
    -------
    Tuple[jnp.ndarray, Optional[jnp.ndarray]]
        (use_key, next_key) where use_key is suitable for a single draw and
        next_key is the remainder of the split.

    Raises
    ------
    ValueError
        If `key` is None.
    """
    if key is None:
        raise ValueError("PRNG key is required; got None")
    use_key, next_key = jax.random.split(key)
    return use_key, next_key
