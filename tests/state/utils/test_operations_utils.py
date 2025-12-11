import jax.numpy as jnp
import pytest

from photon_weave._math.ops import x_operator
from photon_weave.core.meta import make_meta
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


def test_apply_operation_vector_with_shape_plan_matches_full_operator():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    op = x_operator()
    state = _basis_state(0, 4)  # |00>

    out = operations.apply_operation_vector(
        states, targets, state, op, meta=plan
    )
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ state
    assert jnp.allclose(out, expected)


def test_apply_operation_matrix_with_meta_matches_full_operator():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    meta = make_meta((2, 2), (0,))
    op = x_operator()
    state = _basis_state(0, 4)
    rho = state @ jnp.conj(state).T

    # Turn off the global contraction flag to exercise the non-contraction path
    Config().set_contraction(False)
    out = operations.apply_operation_matrix(
        states, targets, rho, op, meta=meta
    )
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out, expected)


def test_apply_operation_matrix_shape_plan_respects_contraction_flag():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    op = x_operator()
    state = _basis_state(0, 4)
    rho = state @ jnp.conj(state).T

    # Force contraction path even when meta is provided
    Config().set_contraction(False)
    out = operations.apply_operation_matrix(
        states, targets, rho, op, meta=plan, use_contraction=True
    )
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out, expected)


def test_apply_kraus_matrix_accepts_shape_plan():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    op = x_operator()
    state = _basis_state(0, 4)
    rho = state @ jnp.conj(state).T

    out = operations.apply_kraus_matrix(states, targets, rho, [op], meta=plan)
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out, expected)


def test_apply_operation_vector_validates_operator_shape():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    bad_op = jnp.eye(3, dtype=jnp.complex128)
    state = _basis_state(0, 4)

    with pytest.raises(AssertionError):
        operations.apply_operation_vector(states, targets, state, bad_op)


def test_contraction_flag_and_default_path_match():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    op = x_operator()
    state = _basis_state(0, 4)

    Config().set_contraction(False)
    out_no_contraction = operations.apply_operation_vector(
        states, targets, state, op, use_contraction=False
    )
    out_contraction = operations.apply_operation_vector(
        states, targets, state, op, use_contraction=True
    )
    assert jnp.allclose(out_no_contraction, out_contraction)
