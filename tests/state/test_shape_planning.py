import jax
import jax.numpy as jnp
import pytest

from photon_weave._math.ops import x_operator
from photon_weave.core import adapters
from photon_weave.photon_weave import Config
from photon_weave.state.utils.shape_planning import (
    build_plan,
    compiled_kernels,
)


class DummyState:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions


def _basis_state(index: int, dim: int) -> jnp.ndarray:
    vec = jnp.zeros((dim, 1), dtype=jnp.complex128)
    return vec.at[index, 0].set(1.0)


@pytest.fixture(autouse=True)
def restore_config_flags():
    cfg = Config()
    prev_contractions = cfg.contractions
    prev_dynamic = cfg.dynamic_dimensions
    prev_jit = cfg.use_jit
    yield
    cfg.set_contraction(prev_contractions)
    cfg.set_dynamic_dimensions(prev_dynamic)
    cfg.set_use_jit(prev_jit)


def test_compiled_kernels_apply_op_vector_matches_full_operator():
    states = [DummyState(2), DummyState(2)]
    plan = build_plan(states, [states[0]])
    kernels = compiled_kernels(plan)
    op = x_operator()
    state = _basis_state(0, 4)  # |00>

    out = kernels.apply_op_vector(state, op)
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ state
    assert jnp.allclose(out, expected)


def test_compiled_kernels_apply_op_matrix_matches_full_operator():
    states = [DummyState(2), DummyState(2)]
    plan = build_plan(states, [states[0]])
    kernels = compiled_kernels(plan)
    op = x_operator()
    state = _basis_state(0, 4)
    rho = state @ jnp.conj(state).T

    Config().set_contraction(True)
    out = kernels.apply_op_matrix(rho, op)
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out, expected)


def test_compiled_kernels_measure_vector_matches_expectation():
    states = [DummyState(2), DummyState(2)]
    plan = build_plan(states, [states[0]])
    kernels = compiled_kernels(plan)
    state = _basis_state(0, 4)  # |00>
    key = jax.random.PRNGKey(0)

    outcome, post, _ = kernels.measure_vector(state, key)
    assert outcome == 0
    assert post.shape == (2, 1)
    assert jnp.allclose(post, _basis_state(0, 2))


def test_compiled_kernels_apply_kraus_matches_full_operator():
    states = [DummyState(2), DummyState(2)]
    plan = build_plan(states, [states[0]])
    kernels = compiled_kernels(plan)
    op = x_operator()
    state = _basis_state(0, 4)
    rho = state @ jnp.conj(state).T
    kraus = jnp.stack([op])

    out = kernels.apply_kraus_matrix(rho, kraus)
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out, expected)


def test_compiled_kernels_trace_out_matches_adapter():
    states = [DummyState(2), DummyState(2), DummyState(2)]
    plan = build_plan(states, [states[0], states[2]])
    kernels = compiled_kernels(plan)
    state = _basis_state(0, 8)
    rho = state @ jnp.conj(state).T

    out = kernels.trace_out_matrix(rho)
    ref = adapters.trace_out_matrix((2, 2, 2), (0, 2), rho)
    assert jnp.allclose(out, ref)


def test_compiled_kernels_vmap_apply_op_vector():
    states = [DummyState(2), DummyState(2)]
    plan = build_plan(states, [states[0]])
    kernels = compiled_kernels(plan)
    op = x_operator()
    batch_states = jnp.stack(
        [
            _basis_state(0, 4).reshape(-1),
            _basis_state(1, 4).reshape(-1),
        ],
        axis=0,
    ).reshape(2, 4, 1)

    vmapped = jax.vmap(kernels.apply_op_vector, in_axes=(0, None))
    out = vmapped(batch_states, op)
    full_op = jnp.kron(op, jnp.eye(2))
    expected = jax.vmap(lambda s: full_op @ s)(batch_states)
    assert jnp.allclose(out, expected)


@pytest.mark.parametrize("use_contraction", [False, True])
def test_compiled_kernels_kraus_and_trace_noncontiguous(use_contraction):
    states = [DummyState(2), DummyState(3), DummyState(2)]
    targets = [states[2]]  # non-contiguous target (index 2)
    plan = build_plan(states, targets)
    kernels = compiled_kernels(plan)
    op = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    kraus = jnp.stack([op])
    state = jnp.zeros((12, 1), dtype=jnp.complex128)
    state = state.at[0, 0].set(1.0)
    rho = state @ jnp.conj(state).T

    out_kraus = kernels.apply_kraus_matrix(rho, kraus)
    full_op = jnp.kron(jnp.eye(3), jnp.kron(jnp.eye(2), op))
    expected_kraus = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out_kraus, expected_kraus)

    # Trace out the middle subsystem, keep targets 0 and 2
    trace_plan = build_plan(states, [states[0], states[2]])
    trace_kernels = compiled_kernels(trace_plan)
    out_trace = trace_kernels.trace_out_matrix(rho)
    ref_trace = adapters.trace_out_matrix((2, 3, 2), (0, 2), rho)
    assert jnp.allclose(out_trace, ref_trace)

    # Vmapped kraus on a small batch
    batch_rho = jnp.stack([rho, rho])
    vmapped = jax.vmap(kernels.apply_kraus_matrix, in_axes=(0, None))
    vmapped_out = vmapped(batch_rho, kraus)
    expected_vmapped = jax.vmap(lambda r: full_op @ r @ jnp.conj(full_op.T))(batch_rho)
    assert jnp.allclose(vmapped_out, expected_vmapped)
