r"""
Adapters to bridge envelope/composite ordering with core kernels.

These helpers reorder flat state tensors so that the target subsystems are
contiguous at the front, call the ordered kernels, and (for operations) restore
the original ordering.
"""

from __future__ import annotations

import math
from functools import lru_cache, partial
from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import opt_einsum as oe

from photon_weave.core import kernels
from photon_weave.core.meta import DimsMeta
from photon_weave.extra.einsum_constructor import (
    apply_operator_matrix as esc_apply_operator_matrix,
)
from photon_weave.extra.einsum_constructor import (
    apply_operator_vector as esc_apply_operator_vector,
)
from photon_weave.extra.einsum_constructor import (
    trace_out_matrix as esc_trace_out_matrix,
)


def _permute_dims(dims: Sequence[int], perm: Sequence[int]) -> List[int]:
    return [dims[i] for i in perm]


def _inverse_perm(perm: Sequence[int]) -> List[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


@lru_cache(None)
def _reorder_meta_vector(
    state_dims: Tuple[int, ...], target_indices: Tuple[int, ...]
) -> Tuple[List[int], List[int], List[int]]:
    """
    Cached permutation metadata for vector reorderings.
    Returns (perm, rest_indices, reordered_dims).
    """
    total_states = len(state_dims)
    rest_indices = [i for i in range(total_states) if i not in target_indices]
    perm = list(target_indices) + rest_indices
    reordered_dims = _permute_dims(state_dims, perm)
    return perm, rest_indices, reordered_dims


@lru_cache(None)
def _reorder_meta_matrix(
    state_dims: Tuple[int, ...], target_indices: Tuple[int, ...]
) -> Tuple[List[int], List[int], List[int]]:
    """
    Cached permutation metadata for matrix reorderings.
    Returns (perm, rest_indices, reordered_dims).
    """
    total_states = len(state_dims)
    rest_indices = [i for i in range(total_states) if i not in target_indices]
    perm = list(target_indices) + rest_indices
    reordered_dims = _permute_dims(state_dims, perm)
    return perm, rest_indices, reordered_dims


@lru_cache(None)
def _cached_trace_out_einsum(
    dims: Tuple[int, ...], target_indices: Tuple[int, ...]
) -> str:
    """
    Cached einsum string for tracing out all but target_indices on a matrix.
    """
    dummy_states = list(range(len(dims)))
    keep = [i for i in dummy_states if i in target_indices]

    # Reuse constructor to avoid rebuilding per call
    class _S:
        def __init__(self, uid):
            self.uid = uid

    state_objs = [_S(i) for i in dummy_states]
    targets = [_S(i) for i in keep]
    return esc_trace_out_matrix(state_objs, targets)


@lru_cache(None)
def _cached_vector_einsum(
    dims: Tuple[int, ...], target_indices: Tuple[int, ...]
) -> str:
    state_axes = list(range(len(dims)))
    target_axes = [state_axes[i] for i in target_indices]
    return esc_apply_operator_vector(state_axes, target_axes)


@lru_cache(None)
def _cached_matrix_einsum(
    dims: Tuple[int, ...], target_indices: Tuple[int, ...]
) -> str:
    state_axes = list(range(len(dims)))
    target_axes = [state_axes[i] for i in target_indices]
    return esc_apply_operator_matrix(state_axes, target_axes)


@lru_cache(None)
def _cached_vector_contraction_kernel(
    state_dims: Tuple[int, ...], target_indices: Tuple[int, ...]
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    einsum_str = _cached_vector_einsum(state_dims, target_indices)
    target_dims = tuple(state_dims[i] for i in target_indices)

    @jax.jit
    def kernel(operator: jnp.ndarray, product_state: jnp.ndarray) -> jnp.ndarray:
        op_tensor = operator.reshape((*target_dims, *target_dims))
        state_tensor = product_state.reshape((*state_dims, 1))
        contracted = oe.contract(einsum_str, op_tensor, state_tensor, backend="jax")
        return contracted.reshape((-1, 1))

    return kernel


@lru_cache(None)
def _cached_matrix_contraction_kernel(
    state_dims: Tuple[int, ...], target_indices: Tuple[int, ...]
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    einsum_str = _cached_matrix_einsum(state_dims, target_indices)
    target_dims = tuple(state_dims[i] for i in target_indices)
    flat_dim = int(math.prod(state_dims))

    @jax.jit
    def kernel(operator: jnp.ndarray, product_state: jnp.ndarray) -> jnp.ndarray:
        op_tensor = operator.reshape((*target_dims, *target_dims))
        state_tensor = product_state.reshape((*state_dims, *state_dims))
        contracted = oe.contract(
            einsum_str,
            op_tensor,
            state_tensor,
            jnp.conj(op_tensor),
            backend="jax",
        )
        return contracted.reshape((flat_dim, flat_dim))

    return kernel


@lru_cache(None)
def _cached_trace_out_contraction_kernel(
    state_dims: Tuple[int, ...], target_indices: Tuple[int, ...]
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    einsum_str = _cached_trace_out_einsum(state_dims, target_indices)
    target_dims = _permute_dims(state_dims, target_indices)
    flat = int(math.prod(target_dims))

    @jax.jit
    def kernel(product_state: jnp.ndarray) -> jnp.ndarray:
        tensor = product_state.reshape((*state_dims, *state_dims))
        reduced = oe.contract(einsum_str, tensor, backend="jax")
        return reduced.reshape((flat, flat))

    return kernel


@lru_cache(None)
def _cached_kraus_einsum(dims: Tuple[int, ...], target_indices: Tuple[int, ...]) -> str:
    """
    Build an einsum string for stacked Kraus operators with a leading Kraus axis.

    Parameters
    ----------
    dims : Tuple[int, ...]
        Dimensions of each subsystem in tensor order.
    target_indices : Tuple[int, ...]
        Indices of the target subsystems the Kraus operators act on.

    Returns
    -------
    str
        An einsum pattern matching operator tensors shaped
        `(k, *target_dims, *target_dims)` and a state tensor shaped
        `(*dims, *dims)`, summing over the shared Kraus axis.

    Examples
    --------
    >>> _cached_kraus_einsum((2, 2), (0,))
    'ka,abcd,kc->kbd'
    """

    base = _cached_matrix_einsum(dims, target_indices)
    lhs, rhs = base.split("->")
    operands = lhs.split(",")

    # Pick a label not used in the base string to represent the Kraus axis.
    used = {c for c in base if c.isalpha()}
    label_ord = ord("k")
    while chr(label_ord) in used:
        label_ord += 1
    kraus_label = chr(label_ord)

    operands[0] = f"{kraus_label}{operands[0]}"
    operands[2] = f"{kraus_label}{operands[2]}"
    return ",".join(operands) + "->" + rhs


def reorder_vector_front(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
) -> Tuple[jnp.ndarray, List[int], List[int]]:
    """
    Move target_indices to the front of the tensorized vector.
    Returns (reordered_state, reordered_dims, permutation_used).
    """
    total_states = len(state_dims)
    perm, rest_indices, reordered_dims = _reorder_meta_vector(
        tuple(state_dims), tuple(target_indices)
    )

    # Fast path: already ordered; avoid reshape/transpose.
    if perm == list(range(total_states)):
        return product_state, list(state_dims), perm

    ps = product_state.reshape((*state_dims, 1))
    ps = jnp.transpose(ps, perm + [total_states])
    ps = ps.reshape((-1, 1))
    return ps, reordered_dims, perm


def restore_vector_order(
    state_dims: Sequence[int],
    perm: Sequence[int],
    product_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Restore vector ordering from a permuted layout back to the original dims.
    """
    if list(perm) == list(range(len(perm))):
        return product_state
    inv = _inverse_perm(perm)
    reordered_dims = _permute_dims(state_dims, perm)
    ps = product_state.reshape((*reordered_dims, 1))
    ps = jnp.transpose(ps, inv + [len(inv)])
    return ps.reshape((math.prod(state_dims), 1))


def reorder_matrix_front(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
) -> Tuple[jnp.ndarray, List[int], List[int]]:
    """
    Move target_indices to the front of the tensorized matrix.
    Returns (reordered_state, reordered_dims, permutation_used).
    """
    total_states = len(state_dims)
    perm, rest_indices, reordered_dims = _reorder_meta_matrix(
        tuple(state_dims), tuple(target_indices)
    )

    # Fast path: already ordered; avoid reshape/transpose.
    if perm == list(range(total_states)):
        return product_state, list(state_dims), perm

    ps = product_state.reshape((*state_dims, *state_dims))
    perm_rows = perm
    perm_cols = [p + total_states for p in perm]
    ps = jnp.transpose(ps, perm_rows + perm_cols)
    return (
        ps.reshape((math.prod(reordered_dims),) * 2),
        reordered_dims,
        perm,
    )


def restore_matrix_order(
    state_dims: Sequence[int],
    perm: Sequence[int],
    product_state: jnp.ndarray,
) -> jnp.ndarray:
    """
    Restore matrix ordering from a permuted layout back to the original dims.
    """
    if list(perm) == list(range(len(perm))):
        return product_state
    inv = _inverse_perm(perm)
    reordered_dims = _permute_dims(state_dims, perm)
    ps = product_state.reshape((*reordered_dims, *reordered_dims))
    total_states = len(inv)
    perm_back = inv + [i + total_states for i in inv]
    ps = jnp.transpose(ps, perm_back)
    return ps.reshape((math.prod(state_dims),) * 2)


def apply_operation_vector(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    """
    Apply an operator to a product-state vector.

    Default: reorder/flatten and call ordered kernels.
    When `use_contraction` is True, apply directly on the tensor via opt_einsum
    to avoid extra reshapes.
    """
    # Fast path: targets already contiguous at the front.
    target_indices = tuple(target_indices)
    state_dims = tuple(state_dims)
    expected = int(math.prod(state_dims))
    if product_state.size != expected:
        raise ValueError(
            f"product_state has size {product_state.size}, expected {expected} "
            f"for dims {state_dims}"
        )
    target_dim = int(math.prod([state_dims[i] for i in target_indices]))
    if operator.shape != (target_dim, target_dim):
        raise AssertionError(
            f"Operator shape {operator.shape} does not match expected "
            f"{(target_dim, target_dim)}"
        )

    if use_contraction:
        return _cached_vector_contraction_kernel(state_dims, target_indices)(
            operator, product_state
        )

    if target_indices == tuple(range(len(target_indices))):
        return kernels.apply_op_vector_ordered(
            state_dims, len(target_indices), product_state, operator
        )

    ps, reordered_dims, perm = reorder_vector_front(
        state_dims, target_indices, product_state
    )
    out = kernels.apply_op_vector_ordered(
        tuple(reordered_dims), len(target_indices), ps, operator
    )
    return restore_vector_order(state_dims, perm, out)


def apply_operation_matrix(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    """
    Apply an operator to a density matrix.

    Default: reorder/flatten and call ordered kernels.
    When `use_contraction` is True, apply directly on the tensor via opt_einsum
    to avoid extra reshapes.
    """
    target_indices = tuple(target_indices)
    state_dims = tuple(state_dims)
    expected = int(math.prod(state_dims))
    if product_state.size != expected * expected:
        raise ValueError(
            f"product_state has size {product_state.size}, expected {expected**2} "
            f"for dims {state_dims}"
        )
    target_dim = int(math.prod([state_dims[i] for i in target_indices]))
    if operator.shape != (target_dim, target_dim):
        raise AssertionError(
            f"Operator shape {operator.shape} does not match expected "
            f"{(target_dim, target_dim)}"
        )

    if use_contraction:
        return _cached_matrix_contraction_kernel(state_dims, target_indices)(
            operator, product_state
        )

    if target_indices == tuple(range(len(target_indices))):
        return kernels.apply_op_matrix_ordered(
            state_dims, len(target_indices), product_state, operator
        )

    ps, reordered_dims, perm = reorder_matrix_front(
        state_dims, target_indices, product_state
    )
    out = kernels.apply_op_matrix_ordered(
        tuple(reordered_dims), len(target_indices), ps, operator
    )
    return restore_matrix_order(state_dims, perm, out)


def apply_kraus_matrix(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    operators: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    """
    Reorder, apply Kraus operators with core kernel, and restore ordering.

    Parameters
    ----------
    state_dims : Sequence[int]
        Dimensions of each subsystem in tensor order.
    target_indices : Sequence[int]
        Indices of the subsystems the Kraus operators act on.
    product_state : jnp.ndarray
        Flattened density matrix shaped ``(prod(dims), prod(dims))``.
    operators : jnp.ndarray
        Stacked Kraus operators shaped ``(k, d, d)`` for the target subsystem.
    use_contraction : bool, optional
        When True, use an einsum contraction path; otherwise reorder and delegate
        to ordered kernels.

    Returns
    -------
    jnp.ndarray
        Updated density matrix with the same flattened shape as ``product_state``.

    Examples
    --------
    Apply a single Kraus operator to the first qubit of a 2-qubit density matrix:

    >>> import jax.numpy as jnp
    >>> rho = jnp.eye(4, dtype=jnp.complex128) / 4
    >>> k = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex128)  # |0><1|
    >>> apply_kraus_matrix((2, 2), (0,), rho, jnp.stack([k]))
    Array([[0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. ],
           [0. , 0. , 0.5, 0. ],
           [0. , 0. , 0. , 0.5]], dtype=complex128)
    """
    target_indices = tuple(target_indices)
    state_dims = tuple(state_dims)
    if use_contraction:
        # operators shape: (k, d, d); reshape to tensor form and contract
        target_dims = [state_dims[i] for i in target_indices]
        op_tensors = operators.reshape((-1, *target_dims, *target_dims))
        state_tensor = product_state.reshape((*state_dims, *state_dims))
        einsum_str = _cached_kraus_einsum(state_dims, target_indices)
        contracted = oe.contract(
            einsum_str,
            op_tensors,
            state_tensor,
            jnp.conj(op_tensors),
            backend="jax",
        )
        flat_dim = int(math.prod(state_dims))
        return contracted.reshape((flat_dim, flat_dim))
    if target_indices == tuple(range(len(target_indices))):
        return kernels.apply_kraus_matrix_ordered(
            state_dims, len(target_indices), product_state, operators
        )

    ps, reordered_dims, perm = reorder_matrix_front(
        state_dims, target_indices, product_state
    )
    out = kernels.apply_kraus_matrix_ordered(
        tuple(reordered_dims), len(target_indices), ps, operators
    )
    return restore_matrix_order(state_dims, perm, out)


def measure_vector(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    Reorder, measure targets with core kernel, and return outcome and post state
    (post state retains only unmeasured subsystems in original order).
    """
    # total_states = len(state_dims)
    target_indices = tuple(target_indices)
    state_dims = tuple(state_dims)
    perm, rest_indices, reordered_dims = _reorder_meta_vector(
        state_dims, target_indices
    )

    # Fast path: targets already at front.
    if target_indices == tuple(range(len(target_indices))):
        outcome, post, key = kernels.measure_vector_ordered(
            state_dims, len(target_indices), product_state, key
        )
        rest_dims = _permute_dims(state_dims, rest_indices)
        return outcome, post.reshape((math.prod(rest_dims), 1)), key

    ps, reordered_dims, _ = reorder_vector_front(
        state_dims, target_indices, product_state
    )
    outcome, post, key = kernels.measure_vector_ordered(
        tuple(reordered_dims), len(target_indices), ps, key
    )
    rest_dims = _permute_dims(state_dims, rest_indices)
    return outcome, post.reshape((math.prod(rest_dims), 1)), key


def measure_matrix(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    Reorder, measure targets with core kernel, and return outcome and post state
    (post state retains only unmeasured subsystems in original order).
    """
    # total_states = len(state_dims)
    target_indices = tuple(target_indices)
    state_dims = tuple(state_dims)
    perm, rest_indices, reordered_dims = _reorder_meta_matrix(
        state_dims, target_indices
    )

    if target_indices == tuple(range(len(target_indices))):
        outcome, post, key = kernels.measure_matrix_ordered(
            state_dims, len(target_indices), product_state, key
        )
        rest_dims = _permute_dims(state_dims, rest_indices)
        return outcome, post.reshape((math.prod(rest_dims),) * 2), key

    ps, reordered_dims, _ = reorder_matrix_front(
        state_dims, target_indices, product_state
    )
    outcome, post, key = kernels.measure_matrix_ordered(
        tuple(reordered_dims), len(target_indices), ps, key
    )
    rest_dims = _permute_dims(state_dims, rest_indices)
    return outcome, post.reshape((math.prod(rest_dims),) * 2), key


def measure_vector_with_probs(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Reorder, measure targets with core kernel, and return outcome, post state,
    probability distribution, and next key.
    """
    # total_states = len(state_dims)
    target_indices = tuple(target_indices)
    state_dims = tuple(state_dims)
    perm, rest_indices, reordered_dims = _reorder_meta_vector(
        state_dims, target_indices
    )

    if target_indices == tuple(range(len(target_indices))):
        outcome, post, probs, key = kernels.measure_vector_ordered_with_probs(
            state_dims, len(target_indices), product_state, key
        )
        rest_dims = _permute_dims(state_dims, rest_indices)
        return outcome, post.reshape((math.prod(rest_dims), 1)), probs, key

    ps, reordered_dims, _ = reorder_vector_front(
        state_dims, target_indices, product_state
    )
    outcome, post, probs, key = kernels.measure_vector_ordered_with_probs(
        tuple(reordered_dims), len(target_indices), ps, key
    )
    rest_dims = _permute_dims(state_dims, rest_indices)
    return outcome, post.reshape((math.prod(rest_dims), 1)), probs, key


def measure_matrix_with_probs(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Reorder, measure targets with core kernel, and return outcome, post state,
    probability distribution, and next key.
    """
    # total_states = len(state_dims)
    target_indices = tuple(target_indices)
    state_dims = tuple(state_dims)
    perm, rest_indices, reordered_dims = _reorder_meta_matrix(
        state_dims, target_indices
    )

    if target_indices == tuple(range(len(target_indices))):
        outcome, post, probs, key = kernels.measure_matrix_ordered_with_probs(
            state_dims, len(target_indices), product_state, key
        )
        rest_dims = _permute_dims(state_dims, rest_indices)
        return (
            outcome,
            post.reshape((math.prod(rest_dims),) * 2),
            probs,
            key,
        )

    ps, reordered_dims, _ = reorder_matrix_front(
        state_dims, target_indices, product_state
    )
    outcome, post, probs, key = kernels.measure_matrix_ordered_with_probs(
        tuple(reordered_dims), len(target_indices), ps, key
    )
    rest_dims = _permute_dims(state_dims, rest_indices)
    return (
        outcome,
        post.reshape((math.prod(rest_dims),) * 2),
        probs,
        key,
    )


def measure_vector_expectation(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reorder, compute measurement probabilities, and return expected
    post-measurement density matrix for the unmeasured subsystems.
    """
    target_indices = tuple(target_indices)
    state_dims = tuple(state_dims)
    _, rest_indices, _ = _reorder_meta_vector(state_dims, target_indices)

    if target_indices == tuple(range(len(target_indices))):
        probs, post = kernels.measure_vector_expectation_ordered(
            state_dims, len(target_indices), product_state
        )
        rest_dims = _permute_dims(state_dims, rest_indices)
        return probs, post.reshape((math.prod(rest_dims),) * 2)

    ps, reordered_dims, _ = reorder_vector_front(
        state_dims, target_indices, product_state
    )
    probs, post = kernels.measure_vector_expectation_ordered(
        tuple(reordered_dims), len(target_indices), ps
    )
    rest_dims = _permute_dims(state_dims, rest_indices)
    return probs, post.reshape((math.prod(rest_dims),) * 2)


def measure_matrix_expectation(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reorder, compute measurement probabilities, and return expected
    post-measurement density matrix for the unmeasured subsystems.
    """
    target_indices = tuple(target_indices)
    state_dims = tuple(state_dims)
    _, rest_indices, _ = _reorder_meta_matrix(state_dims, target_indices)

    if target_indices == tuple(range(len(target_indices))):
        probs, post = kernels.measure_matrix_expectation_ordered(
            state_dims, len(target_indices), product_state
        )
        rest_dims = _permute_dims(state_dims, rest_indices)
        return probs, post.reshape((math.prod(rest_dims),) * 2)

    ps, reordered_dims, _ = reorder_matrix_front(
        state_dims, target_indices, product_state
    )
    probs, post = kernels.measure_matrix_expectation_ordered(
        tuple(reordered_dims), len(target_indices), ps
    )
    rest_dims = _permute_dims(state_dims, rest_indices)
    return probs, post.reshape((math.prod(rest_dims),) * 2)


def measure_povm_matrix(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    operators: jnp.ndarray,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    Reorder, perform POVM with core kernel, return outcome and post state.
    """
    ps, reordered_dims, perm = reorder_matrix_front(
        tuple(state_dims), tuple(target_indices), product_state
    )
    outcome, post, key = kernels.measure_povm_matrix_ordered(
        tuple(reordered_dims), len(target_indices), operators, ps, key
    )
    restored = restore_matrix_order(state_dims, perm, post)
    return outcome, restored, key


def trace_out_matrix(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    """
    Reorder to trace out non-targets and return only the target subsystem(s).
    """
    total_states = len(state_dims)
    rest_indices = [i for i in range(total_states) if i not in target_indices]
    # If all subsystems are requested, return the state unchanged.
    if len(rest_indices) == 0:
        return product_state
    expected = int(math.prod(state_dims))
    if product_state.size != expected * expected:
        raise ValueError(
            f"product_state has size {product_state.size}, expected {expected**2} "
            f"for dims {state_dims}"
        )
    if use_contraction:
        return _cached_trace_out_contraction_kernel(
            tuple(state_dims), tuple(target_indices)
        )(product_state)
    perm = rest_indices + list(target_indices)
    reordered_dims = _permute_dims(state_dims, perm)

    ps = product_state.reshape((*state_dims, *state_dims))
    perm_rows = perm
    perm_cols = [p + total_states for p in perm]
    ps = jnp.transpose(ps, perm_rows + perm_cols)
    ps = ps.reshape((math.prod(reordered_dims),) * 2)

    # Trace out the leading rest_count subsystems (i.e., the ones not requested)
    rest_count = len(rest_indices)
    reduced = kernels.trace_out_matrix_ordered(tuple(reordered_dims), rest_count, ps)
    target_dims = _permute_dims(state_dims, target_indices)
    return reduced.reshape((math.prod(target_dims),) * 2)


@partial(
    jax.jit,
    static_argnames=("state_dims", "target_indices", "use_contraction"),
)
def _apply_operation_vector_jitted(
    state_dims: Tuple[int, ...],
    target_indices: Tuple[int, ...],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return apply_operation_vector(
        state_dims, target_indices, product_state, operator, use_contraction
    )


def apply_operation_vector_jit(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return _apply_operation_vector_jitted(
        tuple(state_dims),
        tuple(target_indices),
        product_state,
        operator,
        use_contraction,
    )


def apply_operation_vector_jit_meta(
    meta: DimsMeta,
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return apply_operation_vector_jit(
        meta.state_dims,
        meta.target_indices,
        product_state,
        operator,
        use_contraction,
    )


@partial(
    jax.jit,
    static_argnames=("state_dims", "target_indices", "use_contraction"),
)
def _apply_operation_matrix_jitted(
    state_dims: Tuple[int, ...],
    target_indices: Tuple[int, ...],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return apply_operation_matrix(
        state_dims, target_indices, product_state, operator, use_contraction
    )


def apply_operation_matrix_jit(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return _apply_operation_matrix_jitted(
        tuple(state_dims),
        tuple(target_indices),
        product_state,
        operator,
        use_contraction,
    )


def apply_operation_matrix_jit_meta(
    meta: DimsMeta,
    product_state: jnp.ndarray,
    operator: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return apply_operation_matrix_jit(
        meta.state_dims,
        meta.target_indices,
        product_state,
        operator,
        use_contraction,
    )


@partial(
    jax.jit,
    static_argnames=("state_dims", "target_indices", "use_contraction"),
)
def _apply_kraus_matrix_jitted(
    state_dims: Tuple[int, ...],
    target_indices: Tuple[int, ...],
    product_state: jnp.ndarray,
    operators: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return apply_kraus_matrix(
        state_dims,
        target_indices,
        product_state,
        operators,
        use_contraction=use_contraction,
    )


def apply_kraus_matrix_jit(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    operators: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return _apply_kraus_matrix_jitted(
        tuple(state_dims),
        tuple(target_indices),
        product_state,
        operators,
        use_contraction,
    )


def apply_kraus_matrix_jit_meta(
    meta: DimsMeta,
    product_state: jnp.ndarray,
    operators: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return apply_kraus_matrix_jit(
        meta.state_dims,
        meta.target_indices,
        product_state,
        operators,
        use_contraction=use_contraction,
    )


@partial(jax.jit, static_argnames=("state_dims", "target_indices"))
def _measure_vector_jitted(
    state_dims: Tuple[int, ...],
    target_indices: Tuple[int, ...],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return measure_vector(state_dims, target_indices, product_state, key)


def measure_vector_jit(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return _measure_vector_jitted(
        tuple(state_dims), tuple(target_indices), product_state, key
    )


def measure_vector_jit_meta(
    meta: DimsMeta, product_state: jnp.ndarray, key: jnp.ndarray
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return measure_vector_jit(meta.state_dims, meta.target_indices, product_state, key)


@partial(jax.jit, static_argnames=("state_dims", "target_indices"))
def _measure_matrix_jitted(
    state_dims: Tuple[int, ...],
    target_indices: Tuple[int, ...],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return measure_matrix(state_dims, target_indices, product_state, key)


def measure_matrix_jit(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return _measure_matrix_jitted(
        tuple(state_dims), tuple(target_indices), product_state, key
    )


def measure_matrix_jit_meta(
    meta: DimsMeta, product_state: jnp.ndarray, key: jnp.ndarray
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return measure_matrix_jit(meta.state_dims, meta.target_indices, product_state, key)


@partial(jax.jit, static_argnames=("state_dims", "target_indices"))
def _measure_povm_matrix_jitted(
    state_dims: Tuple[int, ...],
    target_indices: Tuple[int, ...],
    operators: jnp.ndarray,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return measure_povm_matrix(
        state_dims, target_indices, operators, product_state, key
    )


def measure_povm_matrix_jit(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    operators: jnp.ndarray,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return _measure_povm_matrix_jitted(
        tuple(state_dims), tuple(target_indices), operators, product_state, key
    )


def measure_povm_matrix_jit_meta(
    meta: DimsMeta,
    operators: jnp.ndarray,
    product_state: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    return measure_povm_matrix_jit(
        meta.state_dims, meta.target_indices, operators, product_state, key
    )


@partial(
    jax.jit,
    static_argnames=("state_dims", "target_indices", "use_contraction"),
)
def _trace_out_matrix_jitted(
    state_dims: Tuple[int, ...],
    target_indices: Tuple[int, ...],
    product_state: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return trace_out_matrix(
        state_dims,
        target_indices,
        product_state,
        use_contraction=bool(use_contraction),
    )


def trace_out_matrix_jit(
    state_dims: Sequence[int],
    target_indices: Sequence[int],
    product_state: jnp.ndarray,
    use_contraction: bool = False,
) -> jnp.ndarray:
    return _trace_out_matrix_jitted(
        tuple(state_dims),
        tuple(target_indices),
        product_state,
        bool(use_contraction),
    )


def trace_out_matrix_jit_meta(
    meta: DimsMeta, product_state: jnp.ndarray, use_contraction: bool = False
) -> jnp.ndarray:
    return trace_out_matrix_jit(
        meta.state_dims,
        meta.target_indices,
        product_state,
        use_contraction=use_contraction,
    )
