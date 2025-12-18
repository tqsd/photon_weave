import jax.numpy as jnp
import pytest

from photon_weave._math.ops import x_operator
from photon_weave.photon_weave import Config
from photon_weave.state.utils import operations
from photon_weave.state.utils.shape_planning import build_plan


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


def test_vector_contraction_handles_noncontiguous_target_with_plan():
    states = [DummyState(2), DummyState(2), DummyState(2)]
    targets = [states[1]]  # middle target
    plan = build_plan(states, targets)
    op = x_operator()
    state = _basis_state(2, 8)  # |010>

    out = operations.apply_operation_vector(states, targets, state, op, meta=plan)
    full_op = jnp.kron(jnp.eye(2), jnp.kron(op, jnp.eye(2)))
    expected = full_op @ state
    assert jnp.allclose(out, expected)


def test_matrix_contraction_on_noncontiguous_target_matches_full_op():
    states = [DummyState(2), DummyState(2), DummyState(2)]
    targets = [states[1]]  # middle target
    op = x_operator()
    state = _basis_state(2, 8)  # |010>
    rho = state @ jnp.conj(state).T

    Config().set_contraction(True)
    out = operations.apply_operation_matrix(states, targets, rho, op)
    full_op = jnp.kron(jnp.eye(2), jnp.kron(op, jnp.eye(2)))
    expected = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out, expected)


def test_vector_contraction_with_plan_uses_cached_einsum():
    states = [DummyState(2), DummyState(3)]
    targets = [states[1]]
    plan = build_plan(states, targets)
    op = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=jnp.complex128)
    state = jnp.zeros((6, 1), dtype=jnp.complex128)
    state = state.at[4, 0].set(1.0)  # |10>

    Config().set_contraction(True)
    out = operations.apply_operation_vector(
        states, targets, state, op, meta=plan, use_contraction=True
    )
    full_op = jnp.kron(jnp.eye(2), op)
    expected = full_op @ state
    assert jnp.allclose(out, expected)


def test_matrix_contraction_with_plan_matches_full_operator():
    states = [DummyState(2), DummyState(3)]
    targets = [states[1]]
    plan = build_plan(states, targets)
    op = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=jnp.complex128)
    state = _basis_state(4, 6)  # |10>
    rho = state @ jnp.conj(state).T

    Config().set_contraction(True)
    out = operations.apply_operation_matrix(
        states, targets, rho, op, meta=plan, use_contraction=True
    )
    full_op = jnp.kron(jnp.eye(2), op)
    expected = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out, expected)


def test_apply_kraus_matrix_with_plan_and_contraction():
    states = [DummyState(2), DummyState(2)]
    targets = [states[1]]
    plan = build_plan(states, targets)
    op = x_operator()
    kraus = [op]
    state = _basis_state(1, 4)  # |01>
    rho = state @ jnp.conj(state).T

    Config().set_contraction(True)
    out = operations.apply_kraus_matrix(
        states, targets, rho, kraus, meta=plan, use_contraction=True
    )
    full_op = jnp.kron(jnp.eye(2), op)
    expected = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out, expected)
