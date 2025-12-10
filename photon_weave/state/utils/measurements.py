from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import erf, gammaln

from photon_weave.core import adapters, jitted
from photon_weave.core.linear import measure_matrix as core_measure_matrix
from photon_weave.core.linear import (
    measure_povm_matrix as core_measure_povm_matrix,
)
from photon_weave.core.linear import measure_vector as core_measure_vector
from photon_weave.core.meta import DimsMeta
from photon_weave.core.rng import borrow_key
from photon_weave.photon_weave import Config
from photon_weave.state.interfaces import BaseStateLike as BaseState
from photon_weave.state.utils.shape_planning import ShapePlan


def _ensure_key(key: jnp.ndarray | None) -> jnp.ndarray:
    """
    Convenience helper for imperative paths: use provided key or draw from Config.
    """
    return key if key is not None else Config().random_key


def _borrow_optional(
    key: jnp.ndarray | None,
) -> Tuple[jnp.ndarray, jnp.ndarray | None]:
    """
    Split key when provided; otherwise draw a single-use key from Config.

    Parameters
    ----------
    key : jnp.ndarray or None
        Base key to split. When ``None``, a fresh key is drawn.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray or None]
        ``(use_key, next_key)`` where ``next_key`` is ``None`` when the input key
        was ``None``.
    """
    if key is None:
        return Config().random_key, None
    return jax.random.split(key)


def _key_sequence(
    key: jnp.ndarray | None, count: int
) -> Tuple[List[jnp.ndarray], jnp.ndarray | None]:
    """
    Produce a deterministic list of PRNG keys for non-JIT measurement paths.

    Parameters
    ----------
    key : jnp.ndarray or None
        Base key to split. When ``None``, fresh keys are drawn from ``Config``.
    count : int
        Number of keys to generate.

    Returns
    -------
    Tuple[List[jnp.ndarray], jnp.ndarray or None]
        A list of `count` keys and the leftover key (``None`` when the input key
        was ``None``).

    Notes
    -----
    When a key is provided, we split into ``count + 1`` keys and drop the first
    subkey to reduce correlation between consecutive seeds.
    """
    if key is None:
        cfg = Config()
        keys = [cfg.random_key for _ in range(count)]
        return keys, None
    split = jax.random.split(key, count + 1)
    # Drop the first subkey to reduce correlation between adjacent seeds.
    return list(split[1:]), split[0]


def measure_vector(
    state_objs: Union[List["BaseState"], Tuple["BaseState", ...]],
    target_states: Union[List["BaseState"], Tuple["BaseState", ...]],
    product_state: jnp.ndarray,
    prob_callback: Optional[Callable[[BaseState, jnp.ndarray], None]] = None,
    key: jnp.ndarray | None = None,
) -> Tuple[Dict[BaseState, int], jnp.ndarray]:
    """
    Measure a state vector and return outcomes plus the post-measurement state.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure (collapsed in the order provided).
    product_state : jnp.ndarray
        Flattened state vector shaped ``(prod(dims), 1)``.
    prob_callback : callable, optional
        If provided, called as ``prob_callback(target, probs)`` before sampling
        each target.
    key : jnp.ndarray or None, optional
        Base PRNG key; when None, draws from ``Config().random_key``. Keys are
        split per target.

    Returns
    -------
    dict[BaseState, int]
        Measurement outcomes per target.
    jnp.ndarray
        Collapsed state of the remaining subsystems (flattened).
    """
    state_objs = list(state_objs)
    assert isinstance(product_state, jnp.ndarray)
    dims_arr = jnp.array([s.dimensions for s in state_objs])
    expected_dims = int(jnp.prod(dims_arr))
    assert product_state.shape == (expected_dims, 1)
    shape = list(dims_arr) + [1]
    product_state = product_state.reshape(shape)

    outcomes = {}
    keys, _ = _key_sequence(key, len(target_states))
    for state, use_key in zip(target_states, keys):
        state_index = state_objs.index(state)
        moved = jnp.moveaxis(product_state, state_index, 0)
        probs = jnp.sum(jnp.abs(moved) ** 2, axis=tuple(range(1, moved.ndim)))
        probs = probs / jnp.sum(probs)
        if prob_callback is not None:
            prob_callback(state, probs)
        outcome = jax.random.choice(
            use_key, a=jnp.arange(state.dimensions), p=probs
        )
        outcomes[state] = int(outcome)
        product_state = jnp.take(
            product_state, indices=outcome, axis=state_index
        )
        state_objs.remove(state)
    product_state = product_state.reshape(-1, 1)
    return outcomes, product_state


def measure_vector_jit(
    state_objs: Union[List["BaseState"], Tuple["BaseState", ...]],
    target_states: Union[List["BaseState"], Tuple["BaseState", ...]],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
    meta: DimsMeta | ShapePlan | None = None,
) -> Tuple[Dict[BaseState, int], jnp.ndarray, jnp.ndarray]:
    """
    JIT-friendly measurement wrapper for state vectors.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure.
    product_state : jnp.ndarray
        Flattened state vector shaped ``(prod(dims), 1)``.
    key : jnp.ndarray
        PRNG key; required for JIT.
    meta : DimsMeta or ShapePlan or None, optional
        Static dimension metadata; when provided, routes through jitted adapter
        paths.

    Returns
    -------
    dict[BaseState, int]
        Measurement outcomes per target.
    jnp.ndarray
        Collapsed state of the remaining subsystems (flattened).
    jnp.ndarray
        Next PRNG key.
    """
    use_key, next_key = borrow_key(key)
    if isinstance(meta, ShapePlan):
        dims = list(meta.dims)
        target_indices = list(meta.target_indices)
        meta = meta.meta
    if meta is None:
        dims = [s.dimensions for s in state_objs]
        target_indices = [state_objs.index(s) for s in target_states]
        outcome, post, _ = core_measure_vector(
            dims, target_indices, product_state, use_key
        )
    else:
        dims = list(meta.state_dims)
        target_indices = list(meta.target_indices)
        outcome, post, _ = adapters.measure_vector_jit_meta(
            meta, product_state, use_key
        )
    # Decode flattened outcome into per-target outcomes
    target_dims = [dims[i] for i in target_indices]
    decoded = []
    tmp = outcome
    for dim in reversed(target_dims):
        decoded.append(tmp % dim)
        tmp //= dim
    decoded = list(reversed(decoded))
    outcomes = {ts: int(val) for ts, val in zip(target_states, decoded)}
    return outcomes, post, next_key


def measure_matrix(
    state_objs: Union[List["BaseState"], Tuple["BaseState", ...]],
    target_states: Union[List["BaseState"], Tuple["BaseState", ...]],
    product_state: jnp.ndarray,
    prob_callback: Optional[Callable[[BaseState, jnp.ndarray], None]] = None,
    key: jnp.ndarray | None = None,
) -> Tuple[Dict[BaseState, int], jnp.ndarray]:
    """
    Measure a density matrix and return outcomes plus the reduced post-state.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure.
    product_state : jnp.ndarray
        Flattened density matrix shaped ``(prod(dims), prod(dims))``.
    prob_callback : callable, optional
        If provided, called as ``prob_callback(target, probs)`` before sampling
        each target.
    key : jnp.ndarray or None, optional
        Base PRNG key; when None, draws from ``Config().random_key``. Keys are
        split per target.

    Returns
    -------
    dict[BaseState, int]
        Measurement outcomes per target.
    jnp.ndarray
        Collapsed state of the remaining subsystems (flattened).
    """
    state_objs = list(state_objs)
    assert isinstance(product_state, jnp.ndarray)
    dims_arr = jnp.array([s.dimensions for s in state_objs])
    expected_dims = int(jnp.prod(dims_arr))
    assert product_state.shape == (expected_dims, expected_dims)
    shape = list(dims_arr) * 2
    product_state = product_state.reshape(shape)
    outcomes = {}
    keys, _ = _key_sequence(key, len(target_states))
    for state, use_key in zip(target_states, keys):
        state_index = state_objs.index(state)
        target_dim = state.dimensions
        total_states = len(state_objs)
        rest_indices = [i for i in range(total_states) if i != state_index]
        perm_rows = [state_index] + rest_indices
        perm_cols = [i + total_states for i in perm_rows]
        reordered = jnp.transpose(product_state, perm_rows + perm_cols)
        rest_dim = int(
            jnp.prod(
                jnp.array([state_objs[i].dimensions for i in rest_indices])
            )
            or 1
        )
        tensor = reordered.reshape(
            (target_dim, rest_dim, target_dim, rest_dim)
        )
        probs = jnp.einsum("krkr->k", tensor).real
        probs = probs / jnp.sum(probs)
        if prob_callback is not None:
            prob_callback(state, probs)
        outcome = jax.random.choice(
            use_key, a=jnp.arange(state.dimensions), p=probs
        )
        outcomes[state] = int(outcome)
        total_states = len(state_objs)
        indices: List[Union[slice, int]] = [slice(None)] * len(
            product_state.shape
        )
        indices[state_index] = outcome
        indices[state_index + total_states] = outcome
        product_state = product_state[tuple(indices)]
        state_objs.remove(state)
    if len(state_objs) > 0:
        new_dims = jnp.prod(jnp.array([s.dimensions for s in state_objs]))
        product_state = product_state.reshape((new_dims, new_dims))
    return outcomes, product_state


def measure_matrix_jit(
    state_objs: Union[List["BaseState"], Tuple["BaseState", ...]],
    target_states: Union[List["BaseState"], Tuple["BaseState", ...]],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
    meta: DimsMeta | ShapePlan | None = None,
) -> Tuple[Dict[BaseState, int], jnp.ndarray, jnp.ndarray]:
    """
    JIT-friendly measurement wrapper for density matrices.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure.
    product_state : jnp.ndarray
        Flattened density matrix shaped ``(prod(dims), prod(dims))``.
    key : jnp.ndarray
        PRNG key; required for JIT.
    meta : DimsMeta or ShapePlan or None, optional
        Static dimension metadata; when provided, routes through jitted adapter
        paths.

    Returns
    -------
    dict[BaseState, int]
        Measurement outcomes per target.
    jnp.ndarray
        Collapsed state of the remaining subsystems (flattened).
    jnp.ndarray
        Next PRNG key.
    """
    use_key, next_key = borrow_key(key)
    if isinstance(meta, ShapePlan):
        dims = list(meta.dims)
        target_indices = list(meta.target_indices)
        meta = meta.meta
    if meta is None:
        dims = [s.dimensions for s in state_objs]
        target_indices = [state_objs.index(s) for s in target_states]
        outcome, post, _ = core_measure_matrix(
            dims, target_indices, product_state, use_key
        )
    else:
        dims = list(meta.state_dims)
        target_indices = list(meta.target_indices)
        outcome, post, _ = adapters.measure_matrix_jit_meta(
            meta, product_state, use_key
        )
    target_dims = [dims[i] for i in target_indices]
    decoded = []
    tmp = outcome
    for dim in reversed(target_dims):
        decoded.append(tmp % dim)
        tmp //= dim
    decoded = list(reversed(decoded))
    outcomes = {ts: int(val) for ts, val in zip(target_states, decoded)}
    return outcomes, post, next_key


def measure_vector_jit_with_probs(
    state_objs: Union[List["BaseState"], Tuple["BaseState", ...]],
    target_states: Union[List["BaseState"], Tuple["BaseState", ...]],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
    meta: DimsMeta | ShapePlan | None = None,
) -> Tuple[Dict[BaseState, int], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-friendly vector measurement that also returns sampled probs.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure.
    product_state : jnp.ndarray
        Flattened state vector shaped ``(prod(dims), 1)``.
    key : jnp.ndarray
        PRNG key; required for JIT.
    meta : DimsMeta or ShapePlan or None, optional
        Static dimension metadata; when provided, routes through jitted adapter
        paths.

    Returns
    -------
    dict[BaseState, int]
        Measurement outcomes per target.
    jnp.ndarray
        Collapsed state of the remaining subsystems (flattened).
    jnp.ndarray
        Probability distribution over the measured subsystem.
    jnp.ndarray
        Next PRNG key.
    """
    use_key, next_key = borrow_key(key)
    if isinstance(meta, ShapePlan):
        dims = list(meta.dims)
        target_indices = list(meta.target_indices)
        meta = meta.meta
    if meta is None:
        dims = [s.dimensions for s in state_objs]
        target_indices = [state_objs.index(s) for s in target_states]
        outcome, post, probs, _ = adapters.measure_vector_with_probs(
            dims, target_indices, product_state, use_key
        )
    else:
        dims = list(meta.state_dims)
        target_indices = list(meta.target_indices)
        outcome, post, probs, _ = adapters.measure_vector_with_probs(
            meta.state_dims, meta.target_indices, product_state, use_key
        )
    target_dims = [dims[i] for i in target_indices]
    decoded = []
    tmp = outcome
    for dim in reversed(target_dims):
        decoded.append(tmp % dim)
        tmp //= dim
    decoded = list(reversed(decoded))
    outcomes = {ts: int(val) for ts, val in zip(target_states, decoded)}
    return outcomes, post, probs, next_key


def measure_matrix_jit_with_probs(
    state_objs: Union[List["BaseState"], Tuple["BaseState", ...]],
    target_states: Union[List["BaseState"], Tuple["BaseState", ...]],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
    meta: DimsMeta | ShapePlan | None = None,
) -> Tuple[Dict[BaseState, int], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    JIT-friendly density-matrix measurement that also returns probs.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure.
    product_state : jnp.ndarray
        Flattened density matrix shaped ``(prod(dims), prod(dims))``.
    key : jnp.ndarray
        PRNG key; required for JIT.
    meta : DimsMeta or ShapePlan or None, optional
        Static dimension metadata; when provided, routes through jitted adapter
        paths.

    Returns
    -------
    dict[BaseState, int]
        Measurement outcomes per target.
    jnp.ndarray
        Collapsed state of the remaining subsystems (flattened).
    jnp.ndarray
        Probability distribution over the measured subsystem.
    jnp.ndarray
        Next PRNG key.
    """
    use_key, next_key = borrow_key(key)
    if isinstance(meta, ShapePlan):
        dims = list(meta.dims)
        target_indices = list(meta.target_indices)
        meta = meta.meta
    if meta is None:
        dims = [s.dimensions for s in state_objs]
        target_indices = [state_objs.index(s) for s in target_states]
        outcome, post, probs, _ = adapters.measure_matrix_with_probs(
            dims, target_indices, product_state, use_key
        )
    else:
        dims = list(meta.state_dims)
        target_indices = list(meta.target_indices)
        outcome, post, probs, _ = adapters.measure_matrix_with_probs(
            meta.state_dims, meta.target_indices, product_state, use_key
        )
    target_dims = [dims[i] for i in target_indices]
    decoded = []
    tmp = outcome
    for dim in reversed(target_dims):
        decoded.append(tmp % dim)
        tmp //= dim
    decoded = list(reversed(decoded))
    outcomes = {ts: int(val) for ts, val in zip(target_states, decoded)}
    return outcomes, post, probs, next_key


def measure_vector_expectation(
    state_objs: Union[List["BaseState"], Tuple["BaseState", ...]],
    target_states: Union[List["BaseState"], Tuple["BaseState", ...]],
    product_state: jnp.ndarray,
    meta: DimsMeta | ShapePlan | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Analytic expectation of a projective measurement on a state vector.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure.
    product_state : jnp.ndarray
        Flattened state vector shaped ``(prod(dims), 1)``.
    meta : DimsMeta or ShapePlan or None, optional
        Static dimension metadata; when provided, routes through jitted adapter
        paths.

    Returns
    -------
    jnp.ndarray
        Probabilities for all joint outcomes of the targets.
    jnp.ndarray
        Expected post-measurement density matrix of the unmeasured subsystems.
    """
    if isinstance(meta, ShapePlan):
        dims = list(meta.dims)
        target_indices = list(meta.target_indices)
        meta = meta.meta
    if meta is None:
        dims = [s.dimensions for s in state_objs]
        target_indices = [state_objs.index(s) for s in target_states]
        return adapters.measure_vector_expectation(
            dims, target_indices, product_state
        )
    if Config().use_jit:
        return jitted.measure_vector_expectation(meta, product_state)
    return adapters.measure_vector_expectation(
        meta.state_dims, meta.target_indices, product_state
    )


def measure_matrix_expectation(
    state_objs: Union[List["BaseState"], Tuple["BaseState", ...]],
    target_states: Union[List["BaseState"], Tuple["BaseState", ...]],
    product_state: jnp.ndarray,
    meta: DimsMeta | ShapePlan | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Analytic expectation of a projective measurement on a density matrix.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure.
    product_state : jnp.ndarray
        Flattened density matrix shaped ``(prod(dims), prod(dims))``.
    meta : DimsMeta or ShapePlan or None, optional
        Static dimension metadata; when provided, routes through jitted adapter
        paths.

    Returns
    -------
    jnp.ndarray
        Probabilities for all joint outcomes of the targets.
    jnp.ndarray
        Expected post-measurement density matrix of the unmeasured subsystems.
    """
    if isinstance(meta, ShapePlan):
        dims = list(meta.dims)
        target_indices = list(meta.target_indices)
        meta = meta.meta
    if meta is None:
        dims = [s.dimensions for s in state_objs]
        target_indices = [state_objs.index(s) for s in target_states]
        return adapters.measure_matrix_expectation(
            dims, target_indices, product_state
        )
    if Config().use_jit:
        return jitted.measure_matrix_expectation(meta, product_state)
    return adapters.measure_matrix_expectation(
        meta.state_dims, meta.target_indices, product_state
    )


def measure_POVM_matrix(
    state_objs: Union[List["BaseState"], Tuple["BaseState", ...]],
    target_states: Union[List["BaseState"], Tuple["BaseState", ...]],
    operators: Union[List[jnp.ndarray], Tuple[jnp.ndarray]],
    product_state: jnp.ndarray,
    key: jnp.ndarray | None = None,
    meta: DimsMeta | None = None,
) -> Tuple[int, jnp.ndarray]:
    """
    Perform a POVM measurement on a density matrix.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure with the POVM.
    operators : list[jnp.ndarray] or tuple[jnp.ndarray]
        POVM elements shaped ``(d, d)`` for the target subsystem.
    product_state : jnp.ndarray
        Flattened density matrix shaped ``(prod(dims), prod(dims))``.
    key : jnp.ndarray or None, optional
        Base PRNG key; when None, draws from ``Config().random_key``.
    meta : DimsMeta or None, optional
        Static dimension metadata; when provided, routes through jitted adapter
        paths.

    Returns
    -------
    int
        Index of the POVM element sampled.
    jnp.ndarray
        Post-measurement state (flattened).
    """
    key = _ensure_key(key)
    if meta is None:
        dims = [s.dimensions for s in state_objs]
        target_indices = [state_objs.index(s) for s in target_states]
    else:
        dims = list(meta.state_dims)
        target_indices = list(meta.target_indices)
    target_dim = int(jnp.prod(jnp.array([dims[i] for i in target_indices])))
    expected = (target_dim, target_dim)
    for op in operators:
        if op.shape != expected:
            raise AssertionError(
                f"POVM operator shape {op.shape} does not match expected {expected}"
            )
    ops = jnp.stack(
        [
            op.reshape(
                (jnp.prod(jnp.array([s.dimensions for s in target_states])),)
                * 2
            )
            for op in operators
        ]
    )
    use_key, _ = borrow_key(key)
    if meta is not None:
        outcome, post, _ = adapters.measure_povm_matrix_jit_meta(
            meta, ops, product_state, use_key
        )
    else:
        outcome, post, _ = core_measure_povm_matrix(
            dims, target_indices, ops, product_state, use_key
        )
    return outcome, post


def measure_POVM_matrix_jit(
    state_objs: Union[List[BaseState], Tuple[BaseState, ...]],
    target_states: Union[List[BaseState], Tuple[BaseState, ...]],
    operators: jnp.ndarray,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    JIT-friendly POVM wrapper using core adapters/kernels.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure with the POVM.
    operators : jnp.ndarray
        Stacked POVM elements shaped ``(k, d, d)``.
    product_state : jnp.ndarray
        Flattened density matrix shaped ``(prod(dims), prod(dims))``.
    key : jnp.ndarray
        PRNG key.

    Returns
    -------
    int
        Index of the POVM element sampled.
    jnp.ndarray
        Post-measurement state (flattened).
    jnp.ndarray
        Next PRNG key.
    """
    dims = [s.dimensions for s in state_objs]
    target_indices = [state_objs.index(s) for s in target_states]
    return core_measure_povm_matrix(
        dims, target_indices, operators, product_state, key
    )


def _pnr_noise(
    key: jnp.ndarray | None,
    target_count: int,
    dark_mean: float,
    jitter_std: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample dark counts (Poisson) and timing jitter (Gaussian) for each target.
    """
    key = _ensure_key(key)
    dark_key, jitter_key = jax.random.split(key)
    dark = jax.random.poisson(dark_key, lam=dark_mean, shape=(target_count,))
    if jitter_std > 0:
        jitter = (
            jax.random.normal(jitter_key, shape=(target_count,)) * jitter_std
        )
    else:
        jitter = jnp.zeros((target_count,))
    return dark, jitter


def _decode_flat_outcome(
    outcome: int, target_dims: jnp.ndarray
) -> jnp.ndarray:
    """
    Decode flattened outcome index into per-target outcomes using JAX control flow.
    """

    def body(carry, dim):
        q = carry // dim
        r = carry % dim
        return q, r

    _, outs_rev = lax.scan(body, outcome, target_dims[::-1])
    return outs_rev[::-1]


def measure_pnr_vector(
    state_objs: Union[List["BaseState"], Tuple["BaseState", ...]],
    target_states: Union[List["BaseState"], Tuple["BaseState", ...]],
    product_state: jnp.ndarray,
    efficiency: float = 1.0,
    dark_rate: float = 0.0,
    detection_window: float = 1.0,
    jitter_std: float = 0.0,
    key: jnp.ndarray | None = None,
    meta: DimsMeta | ShapePlan | None = None,
    use_jit: bool | None = None,
) -> Tuple[Dict[BaseState, int], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Photon-number-resolving measurement on a state vector with loss and noise.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure.
    product_state : jnp.ndarray
        Flattened state vector shaped ``(prod(dims), 1)``.
    efficiency : float, optional
        Detection efficiency in [0, 1]; models binomial thinning. Default 1.0.
    dark_rate : float, optional
        Dark count rate (per unit time). Default 0.0.
    detection_window : float, optional
        Detection time window used with `dark_rate`. Default 1.0.
    jitter_std : float, optional
        Standard deviation of Gaussian timing jitter. Default 0.0.
    key : jnp.ndarray or None, optional
        Base PRNG key; required when `use_jit`/`meta` forces the static path.
    meta : DimsMeta or ShapePlan or None, optional
        Static dimension metadata; when provided, routes through jitted adapter
        paths.
    use_jit : bool or None, optional
        Force or disable the jitted path; defaults to `Config().use_jit` when
        not provided.

    Returns
    -------
    dict[BaseState, int]
        Photon counts per target after efficiency and dark counts.
    jnp.ndarray
        Post-measurement state (flattened) of remaining subsystems.
    jnp.ndarray
        Jitter samples per target.
    jnp.ndarray
        Next PRNG key after all sampling steps.

    Model
    -----
    - Efficiency: each detected photon is kept with probability `efficiency`
      (binomial thinning).
    - Dark counts: Poisson process with mean `dark_rate * detection_window`.
    - Jitter: independent Gaussian noise with std `jitter_std` for each target.
    """
    # Jittable path using static dims/indices (via meta or explicit flag)
    use_jit_flag = isinstance(meta, (DimsMeta, ShapePlan)) or (
        Config().use_jit if use_jit is None else use_jit
    )
    if use_jit_flag:
        if isinstance(meta, ShapePlan):
            dims = list(meta.dims)
            target_indices = list(meta.target_indices)
            meta = meta.meta
        if meta is None:
            dims = [s.dimensions for s in state_objs]
            target_indices = [state_objs.index(s) for s in target_states]
        else:
            dims = list(meta.state_dims)
            target_indices = list(meta.target_indices)
        if key is None:
            raise ValueError(
                "key is required for jitted PNR measurement (static path)"
            )
        outcome_flat, post, key_after_meas = core_measure_vector(
            dims, target_indices, product_state, key
        )
        target_dims = jnp.array([dims[i] for i in target_indices])
        decoded = _decode_flat_outcome(outcome_flat, target_dims)
        binom_key, noise_seed, next_key = jax.random.split(
            key_after_meas, 3
        )
        detected = jax.random.binomial(
            binom_key,
            n=decoded,
            p=jnp.clip(efficiency, 0.0, 1.0),
        )
        dark_seed, jitter_seed = jax.random.split(noise_seed)
        dark, jitter = _pnr_noise(
            dark_seed,
            len(target_states),
            dark_rate * detection_window,
            jitter_std,
        )
        totals = detected + dark
        total_outcomes = {
            ts: int(totals[i]) for i, ts in enumerate(target_states)
        }
        return total_outcomes, post, jitter, next_key

    # Fallback (non-jitted) path
    base_key = _ensure_key(key)
    meas_key, noise_key = jax.random.split(base_key)
    outcomes, post = measure_vector(
        state_objs, target_states, product_state, key=meas_key
    )
    binom_key, noise_seed, next_key = jax.random.split(noise_key, 3)
    measured = {}
    subkey = binom_key
    for ts, outcome in outcomes.items():
        subkey, draw_key = jax.random.split(subkey)
        detected = int(
            jax.random.binomial(
                draw_key, n=int(outcome), p=jnp.clip(efficiency, 0.0, 1.0)
            )
        )
        measured[ts] = detected

    dark_seed, jitter_seed = jax.random.split(noise_seed)
    dark, jitter = _pnr_noise(
        dark_seed, len(target_states), dark_rate * detection_window, jitter_std
    )
    total_outcomes: Dict[BaseState, int] = {}
    for idx, ts in enumerate(target_states):
        total_outcomes[ts] = int(measured[ts] + dark[idx])
    return total_outcomes, post, jitter, next_key


def measure_pnr_matrix(
    state_objs: Union[List["BaseState"], Tuple["BaseState", ...]],
    target_states: Union[List["BaseState"], Tuple["BaseState", ...]],
    product_state: jnp.ndarray,
    efficiency: float = 1.0,
    dark_rate: float = 0.0,
    detection_window: float = 1.0,
    jitter_std: float = 0.0,
    key: jnp.ndarray | None = None,
    meta: DimsMeta | ShapePlan | None = None,
    use_jit: bool | None = None,
) -> Tuple[Dict[BaseState, int], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Photon-number-resolving measurement on a density matrix with loss and noise.

    Parameters
    ----------
    state_objs : list[BaseState] or tuple[BaseState, ...]
        All subsystems in tensor order.
    target_states : list[BaseState] or tuple[BaseState, ...]
        Subsystems to measure.
    product_state : jnp.ndarray
        Flattened density matrix shaped ``(prod(dims), prod(dims))``.
    efficiency : float, optional
        Detection efficiency in [0, 1]; models binomial thinning. Default 1.0.
    dark_rate : float, optional
        Dark count rate (per unit time). Default 0.0.
    detection_window : float, optional
        Detection time window used with `dark_rate`. Default 1.0.
    jitter_std : float, optional
        Standard deviation of Gaussian timing jitter. Default 0.0.
    key : jnp.ndarray or None, optional
        Base PRNG key; required when `use_jit`/`meta` forces the static path.
    meta : DimsMeta or ShapePlan or None, optional
        Static dimension metadata; when provided, routes through jitted adapter
        paths.
    use_jit : bool or None, optional
        Force or disable the jitted path; defaults to `Config().use_jit` when
        not provided.

    Returns
    -------
    dict[BaseState, int]
        Photon counts per target after efficiency and dark counts.
    jnp.ndarray
        Post-measurement state (flattened) of remaining subsystems.
    jnp.ndarray
        Jitter samples per target.
    jnp.ndarray
        Next PRNG key after all sampling steps.

    Notes
    -----
    Shares the same physical model as ``measure_pnr_vector`` but operates on
    density matrices.
    """
    use_jit_flag = isinstance(meta, (DimsMeta, ShapePlan)) or (
        Config().use_jit if use_jit is None else use_jit
    )
    if use_jit_flag:
        if isinstance(meta, ShapePlan):
            dims = list(meta.dims)
            target_indices = list(meta.target_indices)
            meta = meta.meta
        if meta is None:
            dims = [s.dimensions for s in state_objs]
            target_indices = [state_objs.index(s) for s in target_states]
        else:
            dims = list(meta.state_dims)
            target_indices = list(meta.target_indices)
        if key is None:
            raise ValueError(
                "key is required for jitted PNR measurement (static path)"
            )
        outcome_flat, post, key_after_meas = core_measure_matrix(
            dims, target_indices, product_state, key
        )
        target_dims = jnp.array([dims[i] for i in target_indices])
        decoded = _decode_flat_outcome(outcome_flat, target_dims)
        binom_key, noise_seed, next_key = jax.random.split(
            key_after_meas, 3
        )
        detected = jax.random.binomial(
            binom_key,
            n=decoded,
            p=jnp.clip(efficiency, 0.0, 1.0),
        )
        dark_seed, jitter_seed = jax.random.split(noise_seed)
        dark, jitter = _pnr_noise(
            dark_seed,
            len(target_states),
            dark_rate * detection_window,
            jitter_std,
        )
        totals = detected + dark
        total_outcomes = {
            ts: int(totals[i]) for i, ts in enumerate(target_states)
        }
        return total_outcomes, post, jitter, next_key

    base_key = _ensure_key(key)
    meas_key, noise_key = jax.random.split(base_key)
    outcomes, post = measure_matrix(
        state_objs, target_states, product_state, key=meas_key
    )
    binom_key, noise_seed, next_key = jax.random.split(noise_key, 3)
    measured = {}
    subkey = binom_key
    for ts, outcome in outcomes.items():
        subkey, draw_key = jax.random.split(subkey)
        detected = int(
            jax.random.binomial(
                draw_key, n=int(outcome), p=jnp.clip(efficiency, 0.0, 1.0)
            )
        )
        measured[ts] = detected
    dark_seed, jitter_seed = jax.random.split(noise_seed)
    dark, jitter = _pnr_noise(
        dark_seed, len(target_states), dark_rate * detection_window, jitter_std
    )
    total_outcomes: Dict[BaseState, int] = {}
    for idx, ts in enumerate(target_states):
        total_outcomes[ts] = int(measured[ts] + dark[idx])
    return total_outcomes, post, jitter, next_key


# ---------------------------------------------------------------------------
# Analytic PNR PMFs/POVMs with efficiency, dark counts, and jitter kernels
# ---------------------------------------------------------------------------


def fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))


def poisson_pmf_direct(k_max: int, mean: float) -> jnp.ndarray:
    k = jnp.arange(k_max + 1, dtype=jnp.float64)
    fact = jnp.exp(gammaln(k + 1))
    pmf = jnp.exp(-mean) * jnp.power(mean, k) / fact
    s = pmf.sum()
    pmf = jnp.where(s > 0, pmf / s, pmf)
    return pmf


def pnr_pmf_single_n(
    n_photons: int,
    eta: float,
    dark_rate: float,
    window: float,
    max_counts: int | None = None,
) -> jnp.ndarray:
    """
    P(K = k | N = n) with binomial efficiency and Poisson dark counts.
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must be in [0, 1].")
    mu_d = dark_rate * window
    n = int(n_photons)
    if max_counts is None:
        mean_sig = n * eta
        var_sig = n * eta * (1.0 - eta)
        mean_tot = mean_sig + mu_d
        var_tot = var_sig + mu_d
        std_tot = jnp.sqrt(jnp.maximum(var_tot, 1e-12))
        cutoff = int(jnp.ceil(mean_tot + 8.0 * std_tot))
        max_counts = int(max(cutoff, n + int(5 + 5 * mu_d)))
    max_counts = max(0, int(max_counts))

    m_vals = jnp.arange(n + 1, dtype=jnp.float64)
    binom_coeffs = jnp.exp(
        gammaln(n + 1) - gammaln(m_vals + 1) - gammaln(n - m_vals + 1)
    )
    P_sig = (
        binom_coeffs
        * jnp.power(eta, m_vals)
        * jnp.power(1.0 - eta, n - m_vals)
    )
    P_dark = poisson_pmf_direct(k_max=max_counts, mean=mu_d)

    m = m_vals[:, None]  # (n+1, 1)
    k = jnp.arange(max_counts + 1)[None, :]  # (1, K)
    valid = m <= k
    k_minus_m = jnp.asarray(k - m, dtype=jnp.int32)
    pdark = jnp.where(valid, jnp.take(P_dark, k_minus_m, mode="clip"), 0.0)
    P_total = (P_sig[:, None] * pdark).sum(axis=0)
    s = P_total.sum()
    P_total = jnp.where(s > 0, P_total / s, P_total)
    return P_total


def pnr_transition_matrix(
    max_photons: int,
    eta: float,
    dark_rate: float,
    window: float,
    max_counts: int | None = None,
) -> jnp.ndarray:
    """
    Classical transition matrix P[k, n] = P(K = k | N = n) for n=0..max_photons.
    """
    max_photons = int(max_photons)
    if max_photons < 0:
        raise ValueError("max_photons must be non-negative.")
    if max_counts is None:
        tmp = pnr_pmf_single_n(
            n_photons=max_photons,
            eta=eta,
            dark_rate=dark_rate,
            window=window,
            max_counts=None,
        )
        max_counts = tmp.shape[0] - 1
    max_counts = int(max_counts)
    cols = []
    for n in range(max_photons + 1):
        cols.append(
            pnr_pmf_single_n(
                n_photons=n,
                eta=eta,
                dark_rate=dark_rate,
                window=window,
                max_counts=max_counts,
            )
        )
    return jnp.stack(cols, axis=1)


def pnr_povm(
    max_photons: int,
    eta: float,
    dark_rate: float,
    window: float,
    max_counts: int | None = None,
) -> jnp.ndarray:
    """
    POVM elements Π_k (diagonal in Fock basis) derived from the transition matrix.
    """
    P = pnr_transition_matrix(
        max_photons=max_photons,
        eta=eta,
        dark_rate=dark_rate,
        window=window,
        max_counts=max_counts,
    )
    max_counts = P.shape[0] - 1
    dim = P.shape[1]
    diag_idx = jnp.arange(dim)
    povm = jnp.zeros((max_counts + 1, dim, dim), dtype=jnp.float64)

    def fill(k, povm_arr):
        return povm_arr.at[k, diag_idx, diag_idx].set(P[k])

    povm = jax.lax.fori_loop(0, max_counts + 1, fill, povm)
    return povm


def gaussian_jitter_kernel(
    num_bins: int,
    bin_width: float,
    jitter_std: float,
) -> jnp.ndarray:
    """
    Discrete Gaussian timing-jitter kernel J[i, j] = P(obs bin i | true bin j).
    """
    if jitter_std <= 0.0:
        return jnp.eye(num_bins, dtype=jnp.float64)
    centers = (jnp.arange(num_bins, dtype=jnp.float64) + 0.5) * bin_width
    sqrt2_sigma = jnp.sqrt(2.0) * jitter_std

    def column(t0):
        def entry(ti):
            left = (ti - 0.5 * bin_width - t0) / sqrt2_sigma
            right = (ti + 0.5 * bin_width - t0) / sqrt2_sigma
            return 0.5 * (erf(right) - erf(left))

        col = jax.vmap(entry)(centers)
        s = col.sum()
        return jnp.where(s > 0, col / s, col)

    return jax.vmap(column)(centers).T
