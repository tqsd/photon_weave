"""
Stateless tensor kernels intended for JAX JIT/vmap.

Design notes
------------
- Kernels operate purely on tensors plus lightweight metadata (state_dims,
  target_count) and do not touch envelope/composite state.
- Callers are responsible for ordering the target subsystem axes to the front
  (see adapters in `photon_weave.core.adapters`) and reshaping flat states.
- Shapes should be static per compiled instance; if dimensions need to change,
  handle that outside the jitted path (e.g., by padding or recompiling).
"""

from __future__ import annotations

from functools import reduce
from typing import Iterable, Sequence, Tuple

import jax
import jax.numpy as jnp
import opt_einsum as oe


def _prod(values: Iterable[int]) -> int:
    return int(reduce(lambda a, b: a * b, values, 1))


def _measurement_probs_vector(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    target_dims = state_dims[:target_count]
    rest_dims = state_dims[target_count:]
    target_flat = _prod(target_dims)
    rest_flat = _prod(rest_dims)

    ps = product_state.reshape((target_flat, rest_flat, 1))
    probs = oe.contract("tcr->t", jnp.abs(ps) ** 2, backend="jax")
    probs = probs / jnp.sum(probs)
    return probs, ps, rest_flat


def _measurement_probs_matrix(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    target_dims = state_dims[:target_count]
    rest_dims = state_dims[target_count:]
    target_flat = _prod(target_dims)
    rest_flat = _prod(rest_dims)

    ps = product_state.reshape(
        (target_flat, rest_flat, target_flat, rest_flat)
    )
    ps = jnp.transpose(ps, (0, 2, 1, 3))  # (t, t, r, r)
    reduced_target = oe.contract("ijrr->ij", ps, backend="jax")
    probs = jnp.real(jnp.diag(reduced_target))
    probs = probs / jnp.sum(probs)
    return probs, ps, rest_flat


def apply_op_vector_ordered(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply an operator to the leading `target_count` axes of a state vector.

    Parameters
    ----------
    state_dims : Sequence[int]
        Dimensions of each subsystem in tensor order (targets first).
    target_count : int
        Number of leading subsystems the operator acts on.
    product_state : jnp.ndarray
        State vector shaped (prod(state_dims), 1).
    operator : jnp.ndarray
        Operator shaped (prod(target_dims), prod(target_dims)).

    Returns
    -------
    jnp.ndarray
        Updated state vector shaped (prod(state_dims), 1).
    """
    target_dims = state_dims[:target_count]
    rest_dims = state_dims[target_count:]

    target_flat = _prod(target_dims)
    rest_flat = _prod(rest_dims)
    total = target_flat * rest_flat

    ps = product_state.reshape((target_flat, rest_flat, 1))
    op = operator.reshape((target_flat, target_flat))

    ps = oe.contract("ab,bcn->acn", op, ps, backend="jax")
    return ps.reshape((total, 1))


def apply_op_matrix_ordered(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply an operator to the leading `target_count` axes of a density matrix.

    Parameters
    ----------
    state_dims : Sequence[int]
        Dimensions of each subsystem in tensor order (targets first).
    target_count : int
        Number of leading subsystems the operator acts on.
    product_state : jnp.ndarray
        Density matrix shaped (prod(state_dims), prod(state_dims)).
    operator : jnp.ndarray
        Operator shaped (prod(target_dims), prod(target_dims)).

    Returns
    -------
    jnp.ndarray
        Updated density matrix shaped (prod(state_dims), prod(state_dims)).
    """
    target_dims = state_dims[:target_count]
    rest_dims = state_dims[target_count:]

    target_flat = _prod(target_dims)
    rest_flat = _prod(rest_dims)
    total = target_flat * rest_flat

    ps = product_state.reshape(
        (target_flat, rest_flat, target_flat, rest_flat)
    )
    # Bring target axes together for a clean matmul: (t1, t2, r1, r2)
    ps = jnp.transpose(ps, (0, 2, 1, 3)).reshape(
        (target_flat, target_flat, rest_flat * rest_flat)
    )

    op = operator.reshape((target_flat, target_flat))
    tmp = oe.contract("ab,bcn->acn", op, ps, backend="jax")
    out = oe.contract("acn,bc->abn", tmp, jnp.conj(op), backend="jax")

    out = out.reshape((target_flat, target_flat, rest_flat, rest_flat))
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape((total, total))
    return out


def apply_kraus_matrix_ordered(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
    operators: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply stacked Kraus operators (shape [k, d, d]) to leading `target_count`.
    Operators are assumed to already act on the target subsystem only.

    Parameters
    ----------
    state_dims : Sequence[int]
        Dimensions of each subsystem in tensor order (targets first).
    target_count : int
        Number of leading subsystems the Kraus operators act on.
    product_state : jnp.ndarray
        Density matrix shaped (prod(state_dims), prod(state_dims)).
    operators : jnp.ndarray
        Stacked Kraus operators shaped (k, prod(target_dims), prod(target_dims)).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> rho = jnp.eye(4, dtype=jnp.complex128) / 4  # 2-qubit maximally mixed
    >>> k = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex128)
    >>> apply_kraus_matrix_ordered((2, 2), 1, rho, jnp.stack([k]))
    Array([[0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. ],
           [0. , 0. , 0.5, 0. ],
           [0. , 0. , 0. , 0.5]], dtype=complex128)
    """

    def _single(op, ps):
        return apply_op_matrix_ordered(state_dims, target_count, ps, op)

    return jnp.sum(
        jax.vmap(_single, in_axes=(0, None))(operators, product_state), axis=0
    )


def measure_vector_ordered(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    Measure the leading `target_count` axes of a state vector.

    Returns (outcome_index, post_state, next_key) where post_state contains
    only the unmeasured subsystem.
    """
    probs, ps, rest_flat = _measurement_probs_vector(
        state_dims, target_count, product_state
    )
    outcome = jax.random.choice(key, a=ps.shape[0], p=probs)
    post = ps[outcome] / jnp.sqrt(probs[outcome] + 1e-18)
    return outcome, post.reshape((rest_flat, 1)), key


def measure_matrix_ordered(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    Measure the leading `target_count` axes of a density matrix.

    Returns (outcome_index, post_state, next_key) where post_state contains
    only the unmeasured subsystem.
    """
    probs, ps, rest_flat = _measurement_probs_matrix(
        state_dims, target_count, product_state
    )
    outcome = jax.random.choice(key, a=ps.shape[0], p=probs)
    post = ps[outcome, outcome] / (probs[outcome] + 1e-18)
    post = post.reshape((rest_flat, rest_flat))
    return outcome, post, key


def measure_vector_ordered_with_probs(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Measure vector state, returning sampled outcome, post state, probability
    distribution, and next key.
    """
    probs, ps, rest_flat = _measurement_probs_vector(
        state_dims, target_count, product_state
    )
    outcome = jax.random.choice(key, a=ps.shape[0], p=probs)
    post = ps[outcome] / jnp.sqrt(probs[outcome] + 1e-18)
    return outcome, post.reshape((rest_flat, 1)), probs, key


def measure_matrix_ordered_with_probs(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Measure density matrix, returning sampled outcome, post state, probability
    distribution, and next key.
    """
    probs, ps, rest_flat = _measurement_probs_matrix(
        state_dims, target_count, product_state
    )
    outcome = jax.random.choice(key, a=ps.shape[0], p=probs)
    post = ps[outcome, outcome] / (probs[outcome] + 1e-18)
    return outcome, post.reshape((rest_flat, rest_flat)), probs, key


def measure_vector_expectation_ordered(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Deterministic expectation for measuring the leading `target_count` axes of a
    state vector. Returns (probs, expected post-measurement density matrix of the
    unmeasured subsystem).
    """
    probs, ps, rest_flat = _measurement_probs_vector(
        state_dims, target_count, product_state
    )
    norm_states = ps[..., 0] / jnp.sqrt(probs[:, None] + 1e-18)
    expected_post = oe.contract(
        "tr,ts,t->rs", norm_states, jnp.conj(norm_states), probs, backend="jax"
    )
    return probs, expected_post.reshape((rest_flat, rest_flat))


def measure_matrix_expectation_ordered(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Deterministic expectation for measuring the leading `target_count` axes of a
    density matrix. Returns (probs, expected post-measurement density matrix of
    the unmeasured subsystem).
    """
    probs, _, _ = _measurement_probs_matrix(
        state_dims, target_count, product_state
    )
    expected_post = trace_out_matrix_ordered(
        state_dims, target_count, product_state
    )
    return probs, expected_post


def trace_out_matrix_ordered(
    state_dims: Sequence[int],
    target_count: int,
    product_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Trace out the leading `target_count` axes from a density matrix.
    """
    target_dims = state_dims[:target_count]
    rest_dims = state_dims[target_count:]
    target_flat = _prod(target_dims)
    rest_flat = _prod(rest_dims)

    ps = product_state.reshape(
        (target_flat, rest_flat, target_flat, rest_flat)
    )
    ps = jnp.transpose(ps, (0, 2, 1, 3)).reshape(
        (target_flat, target_flat, rest_flat * rest_flat)
    )
    traced = oe.contract("iin->n", ps, backend="jax")
    return traced.reshape((rest_flat, rest_flat))


def measure_povm_matrix_ordered(
    state_dims: Sequence[int],
    target_count: int,
    operators: jnp.ndarray,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    Perform POVM on the leading `target_count` axes. Operators shape [k, d, d].
    Returns (outcome_index, post_state, next_key).
    """

    def _apply(op, ps):
        return apply_op_matrix_ordered(state_dims, target_count, ps, op)

    projected = jax.vmap(_apply, in_axes=(0, None))(operators, product_state)
    probs = jnp.real(oe.contract("kii->k", projected, backend="jax"))
    probs = probs / jnp.sum(probs)

    outcome = jax.random.choice(key, a=operators.shape[0], p=probs)
    post = projected[outcome]
    post = post / (jnp.trace(post) + 1e-18)
    return outcome, post, key


def vmap_apply_op_matrix_ordered(
    state_dims: Sequence[int], target_count: int, operator: jnp.ndarray
):
    """
    Return a vmapped apply_op_matrix_ordered over a batch of product states.
    """
    return jax.vmap(
        lambda ps: apply_op_matrix_ordered(
            state_dims, target_count, ps, operator
        ),
        in_axes=0,
    )
