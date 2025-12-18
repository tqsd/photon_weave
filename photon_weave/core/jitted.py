"""
Jitted entry points for core operations using static dimension metadata.

These helpers keep shapes static for JIT/vmap by requiring a `DimsMeta`
instance. They wrap the adapter-level jitted functions so envelope/state code
can opt in without recomputing dimension indices each call.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from photon_weave.core import adapters
from photon_weave.core.meta import DimsMeta


def apply_operation_vector(
    meta: DimsMeta,
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return adapters.apply_operation_vector_jit_meta(
        meta, product_state, operator, use_contraction
    )


def apply_operation_matrix(
    meta: DimsMeta,
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return adapters.apply_operation_matrix_jit_meta(
        meta, product_state, operator, use_contraction
    )


def apply_kraus_matrix(
    meta: DimsMeta,
    product_state: jnp.ndarray,
    operators: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return adapters.apply_kraus_matrix_jit_meta(
        meta, product_state, operators, use_contraction=use_contraction
    )


def measure_vector(
    meta: DimsMeta, product_state: jnp.ndarray, key: jnp.ndarray
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return adapters.measure_vector_jit_meta(meta, product_state, key)


def measure_matrix(
    meta: DimsMeta, product_state: jnp.ndarray, key: jnp.ndarray
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return adapters.measure_matrix_jit_meta(meta, product_state, key)


def measure_povm_matrix(
    meta: DimsMeta,
    operators: jnp.ndarray,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return adapters.measure_povm_matrix_jit_meta(meta, operators, product_state, key)


def trace_out_matrix(
    meta: DimsMeta, product_state: jnp.ndarray, use_contraction: bool = False
) -> jnp.ndarray:
    return adapters.trace_out_matrix_jit_meta(
        meta, product_state, use_contraction=use_contraction
    )


def measure_vector_with_probs(
    meta: DimsMeta, product_state: jnp.ndarray, key: jnp.ndarray
) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return adapters.measure_vector_with_probs(
        meta.state_dims, meta.target_indices, product_state, key
    )


def measure_matrix_with_probs(
    meta: DimsMeta, product_state: jnp.ndarray, key: jnp.ndarray
) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return adapters.measure_matrix_with_probs(
        meta.state_dims, meta.target_indices, product_state, key
    )


def measure_vector_expectation(
    meta: DimsMeta, product_state: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return adapters.measure_vector_expectation(
        meta.state_dims, meta.target_indices, product_state
    )


def measure_matrix_expectation(
    meta: DimsMeta, product_state: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return adapters.measure_matrix_expectation(
        meta.state_dims, meta.target_indices, product_state
    )
