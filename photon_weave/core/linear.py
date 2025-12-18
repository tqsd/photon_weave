"""
Stateless linear-algebra helpers that wrap adapter/kernels.
Keeping these in a single module prevents import scattering.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax.numpy as jnp

from photon_weave.core import adapters


def apply_operation_vector(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    key: jnp.ndarray | None = None,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return adapters.apply_operation_vector(
        state_dims,
        target_indices,
        product_state,
        operator,
        use_contraction=use_contraction,
    )


def apply_operation_matrix(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    key: jnp.ndarray | None = None,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return adapters.apply_operation_matrix(
        state_dims,
        target_indices,
        product_state,
        operator,
        use_contraction=use_contraction,
    )


def measure_vector(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return adapters.measure_vector(state_dims, target_indices, product_state, key)


def measure_matrix(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return adapters.measure_matrix(state_dims, target_indices, product_state, key)


def measure_povm_matrix(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    operators: jnp.ndarray,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return adapters.measure_povm_matrix(
        state_dims, target_indices, operators, product_state, key
    )


def trace_out_matrix(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return adapters.trace_out_matrix(
        state_dims,
        target_indices,
        product_state,
        use_contraction=use_contraction,
    )


def apply_kraus_matrix(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    operators: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return adapters.apply_kraus_matrix(
        state_dims,
        target_indices,
        product_state,
        operators,
        use_contraction=use_contraction,
    )
