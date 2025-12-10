from __future__ import annotations

from typing import List, Tuple, Union

import jax.numpy as jnp

from photon_weave.core import adapters
from photon_weave.core.linear import trace_out_matrix as core_trace_out_matrix
from photon_weave.core.meta import DimsMeta
from photon_weave.photon_weave import Config
from photon_weave.state.base_state import BaseState
from photon_weave.state.expansion_levels import ExpansionLevel

from .state_transform import state_expand


def trace_out_vector(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
) -> jnp.ndarray:
    """(DEPRECATED) Trace out everything but the target states from a vector."""
    dims = int(jnp.prod(jnp.array([s.dimensions for s in state_objs])))
    product_state, _ = state_expand(product_state, ExpansionLevel.Vector, dims)
    return trace_out_matrix(state_objs, target_states, product_state)


def trace_out_matrix(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
    meta: DimsMeta | None = None,
    use_contraction: bool | None = None,
) -> jnp.ndarray:
    """
    Trace out everything but the target states from a density matrix.
    """
    if use_contraction is None:
        use_contraction = Config().contractions
    if meta is None:
        dims = [s.dimensions for s in state_objs]
        target_indices = [state_objs.index(s) for s in target_states]
        return core_trace_out_matrix(
            dims,
            target_indices,
            product_state,
            use_contraction=bool(use_contraction),
        )
    return adapters.trace_out_matrix_jit_meta(
        meta, product_state, use_contraction=bool(use_contraction)
    )


def trace_out_matrix_jit(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    product_state: jnp.ndarray,
    use_contraction: bool | None = None,
) -> jnp.ndarray:
    """
    JIT-friendly trace-out wrapper using core adapters/kernels.
    """
    if use_contraction is None:
        use_contraction = Config().contractions
    dims = [s.dimensions for s in state_objs]
    target_indices = [state_objs.index(s) for s in target_states]
    return adapters.trace_out_matrix(
        dims,
        target_indices,
        product_state,
        use_contraction=bool(use_contraction),
    )
