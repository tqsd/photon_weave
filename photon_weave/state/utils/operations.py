from __future__ import annotations

from functools import lru_cache
from math import prod
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import opt_einsum as oe

from photon_weave.core import adapters, jitted
from photon_weave.core.linear import (
    apply_operation_matrix as core_apply_operation_matrix,
)
from photon_weave.core.linear import (
    apply_operation_vector as core_apply_operation_vector,
)
from photon_weave.core.meta import DimsMeta
from photon_weave.extra.einsum_constructor import (
    apply_operator_matrix as esc_apply_operator_matrix,
)
from photon_weave.extra.einsum_constructor import (
    apply_operator_vector as esc_apply_operator_vector,
)
from photon_weave.photon_weave import Config
from photon_weave.state.interfaces import BaseStateLike as BaseState
from photon_weave.state.utils.shape_planning import ShapePlan


def _extract_meta(
    meta: DimsMeta | ShapePlan | None,
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
) -> tuple[Tuple[int, ...], Tuple[int, ...], DimsMeta | None]:
    """Normalize dims/target_indices/meta handling for the apply helpers."""
    if isinstance(meta, ShapePlan):
        return meta.dims, meta.target_indices, meta.meta
    if meta is None:
        idx_map: Dict[BaseState, int] = {s: i for i, s in enumerate(state_objs)}
        return (
            tuple(s.dimensions for s in state_objs),
            tuple(idx_map[s] for s in target_states),
            None,
        )
    return meta.state_dims, meta.target_indices, meta


@lru_cache(None)
def _cached_vector_einsum_from_dims(
    dims: Tuple[int, ...], target_indices: Tuple[int, ...]
) -> str:
    """Einsum pattern for vector apply (cached)."""
    state_axes = tuple(range(len(dims)))
    target_axes = tuple(state_axes[i] for i in target_indices)
    return esc_apply_operator_vector(list(state_axes), list(target_axes))


@lru_cache(None)
def _cached_matrix_einsum_from_dims(
    dims: Tuple[int, ...], target_indices: Tuple[int, ...]
) -> str:
    """Einsum pattern for matrix apply (cached)."""
    state_axes = tuple(range(len(dims)))
    target_axes = tuple(state_axes[i] for i in target_indices)
    return esc_apply_operator_matrix(list(state_axes), list(target_axes))


@lru_cache(None)
def _cached_vector_contract_expr(
    dims: Tuple[int, ...], target_indices: Tuple[int, ...]
):
    """Pre-compiled opt_einsum expression for vector apply."""
    target_dims = tuple(dims[i] for i in target_indices)
    op_shape = (*target_dims, *target_dims)
    state_shape = (*dims, 1)
    einsum_str = _cached_vector_einsum_from_dims(dims, target_indices)
    return oe.contract_expression(einsum_str, op_shape, state_shape)


@lru_cache(None)
def _cached_matrix_contract_expr(
    dims: Tuple[int, ...], target_indices: Tuple[int, ...]
):
    """Pre-compiled opt_einsum expression for matrix apply."""
    target_dims = tuple(dims[i] for i in target_indices)
    op_shape = (*target_dims, *target_dims)
    state_shape = (*dims, *dims)
    einsum_str = _cached_matrix_einsum_from_dims(dims, target_indices)
    return oe.contract_expression(einsum_str, op_shape, state_shape, op_shape)


def apply_operation_vector(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    meta: DimsMeta | ShapePlan | None = None,
    key: jnp.ndarray | None = None,
    use_contraction: Optional[bool] = None,
    target_indices: Tuple[int, ...] | None = None,
) -> jnp.ndarray:
    """
    Apply an operator to a state vector using core adapters.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems the operator acts on.
    product_state : jnp.ndarray
        State vector flattened to shape ``(prod(dims), 1)``.
    operator : jnp.ndarray
        Operator shaped ``(d, d)`` for the target subsystem.
    meta : DimsMeta or ShapePlan or None, optional
        Static dimension metadata; when provided the call is routed through
        jitted adapter paths.
    key : jnp.ndarray or None, optional
        Unused in the deterministic operator application path; included for API
        parity with lower-level kernels.
    use_contraction : bool or None, optional
        When True, use an einsum contraction path; when None, default to the
        current ``Config().contractions`` flag.

    Returns
    -------
    jnp.ndarray
        Updated state vector with the same flattened shape as ``product_state``.
    """
    use_contraction = (
        Config().contractions if use_contraction is None else use_contraction
    )
    if target_indices is not None:
        dims, _, meta_for_jit = _extract_meta(meta, state_objs, target_states)
    else:
        dims, target_indices, meta_for_jit = _extract_meta(
            meta, state_objs, target_states
        )
    target_dim = int(prod(dims[i] for i in target_indices))
    if operator.shape != (target_dim, target_dim):
        raise AssertionError(
            f"Operator shape {operator.shape} does not match expected "
            f"{(target_dim, target_dim)}"
        )
    if use_contraction:
        # Prefer jitted path when static meta is available
        if meta_for_jit is not None:
            return jitted.apply_operation_vector(
                meta_for_jit,
                product_state,
                operator,
                use_contraction=True,
            )
        target_dims = [dims[i] for i in target_indices]
        op_tensor = operator.reshape((*target_dims, *target_dims))
        state_tensor = product_state.reshape((*dims, 1))
        expr = _cached_vector_contract_expr(dims, target_indices)
        contracted = expr(op_tensor, state_tensor, backend="jax")
        return contracted.reshape((-1, 1))
    if meta_for_jit is not None:
        return jitted.apply_operation_vector(
            meta_for_jit,
            product_state,
            operator,
            use_contraction=use_contraction,
        )
    return core_apply_operation_vector(
        dims,
        target_indices,
        product_state,
        operator,
        key,
        use_contraction=use_contraction,
    )


def apply_operation_matrix(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    meta: DimsMeta | ShapePlan | None = None,
    key: jnp.ndarray | None = None,
    use_contraction: Optional[bool] = None,
    target_indices: Tuple[int, ...] | None = None,
) -> jnp.ndarray:
    """
    Apply an operator to a density matrix using core adapters.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems the operator acts on.
    product_state : jnp.ndarray
        Density matrix flattened to shape ``(prod(dims), prod(dims))``.
    operator : jnp.ndarray
        Operator shaped ``(d, d)`` for the target subsystem.
    meta : DimsMeta or ShapePlan or None, optional
        Static dimension metadata; when provided the call is routed through
        jitted adapter paths.
    key : jnp.ndarray or None, optional
        Unused in the deterministic operator application path; included for API
        parity with lower-level kernels.
    use_contraction : bool or None, optional
        When True, use an einsum contraction path; when None, default to the
        current ``Config().contractions`` flag.

    Returns
    -------
    jnp.ndarray
        Updated density matrix with the same flattened shape as ``product_state``.
    """
    use_contraction = (
        Config().contractions if use_contraction is None else use_contraction
    )
    if target_indices is not None:
        dims, _, meta_for_jit = _extract_meta(meta, state_objs, target_states)
    else:
        dims, target_indices, meta_for_jit = _extract_meta(
            meta, state_objs, target_states
        )
    target_dim = int(prod(dims[i] for i in target_indices))
    if operator.shape != (target_dim, target_dim):
        raise AssertionError(
            f"Operator shape {operator.shape} does not match expected "
            f"{(target_dim, target_dim)}"
        )
    if use_contraction:
        if meta_for_jit is not None:
            return jitted.apply_operation_matrix(
                meta_for_jit,
                product_state,
                operator,
                use_contraction=True,
            )
        target_dims = [dims[i] for i in target_indices]
        op_tensor = operator.reshape((*target_dims, *target_dims))
        state_tensor = product_state.reshape((*dims, *dims))
        expr = _cached_matrix_contract_expr(dims, target_indices)
        contracted = expr(op_tensor, state_tensor, jnp.conj(op_tensor), backend="jax")
        return contracted.reshape((prod(dims),) * 2)
    if meta_for_jit is not None:
        return jitted.apply_operation_matrix(
            meta_for_jit,
            product_state,
            operator,
            use_contraction=use_contraction,
        )
    return core_apply_operation_matrix(
        dims,
        target_indices,
        product_state,
        operator,
        key,
        use_contraction=use_contraction,
    )


def apply_kraus_matrix(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
    operators: Union[List[jnp.ndarray], Tuple[jnp.ndarray, ...]],
    meta: DimsMeta | ShapePlan | None = None,
    use_contraction: bool | None = None,
    target_indices: Tuple[int, ...] | None = None,
) -> jnp.ndarray:
    """
    Apply stacked Kraus operators to a density matrix.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems the Kraus operators act on.
    product_state : jnp.ndarray
        Density matrix flattened to shape ``(prod(dims), prod(dims))``.
    operators : list[jnp.ndarray] or tuple[jnp.ndarray, ...]
        Kraus operators shaped ``(d, d)`` for the target subsystem; they
        are stacked along a leading axis internally.
    meta : DimsMeta or ShapePlan or None, optional
        Static dimension metadata; when provided the call is routed through
        jitted adapter paths.
    use_contraction : bool or None, optional
        When True, use an einsum contraction path; when None, default to the
        current ``Config().contractions`` flag.

    Returns
    -------
    jnp.ndarray
        Updated density matrix with the same flattened shape as ``product_state``.
    """
    if use_contraction is None:
        use_contraction = Config().contractions
    if isinstance(meta, ShapePlan):
        dims = meta.dims
        target_indices = (
            meta.target_indices if target_indices is None else target_indices
        )
        meta_for_jit = meta.meta
    elif meta is None:
        dims = tuple(s.dimensions for s in state_objs)
        if target_indices is None:
            idx_map: Dict[BaseState, int] = {s: i for i, s in enumerate(state_objs)}
            target_indices = tuple(idx_map[s] for s in target_states)
        meta_for_jit = None
    else:
        dims = meta.state_dims
        target_indices = (
            meta.target_indices if target_indices is None else target_indices
        )
        meta_for_jit = meta
    target_dim = int(prod(dims[i] for i in target_indices))
    expected_shape = (target_dim, target_dim)
    for op in operators:
        if op.shape != expected_shape:
            raise AssertionError(
                "Kraus operator shape "
                f"{op.shape} does not match expected {expected_shape}"
            )
    stacked = jnp.stack([jnp.asarray(op).reshape(expected_shape) for op in operators])
    if meta_for_jit is not None:
        return jitted.apply_kraus_matrix(
            meta_for_jit,
            product_state,
            stacked,
            use_contraction=bool(use_contraction),
        )
    return adapters.apply_kraus_matrix(
        dims,
        target_indices,
        product_state,
        stacked,
        use_contraction=bool(use_contraction),
    )
