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


def test_contraction_flag_enables_tensor_path_without_use_contraction():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    op = x_operator()
    state = _basis_state(0, 4)  # |00>

    Config().set_contraction(True)
    out = operations.apply_operation_vector(
        states, targets, state, op, use_contraction=False
    )
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ state
    assert jnp.allclose(out, expected)


def test_contraction_flag_with_shape_plan_routes_through_jit_path():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    op = x_operator()
    state = _basis_state(0, 4)
    rho = state @ jnp.conj(state).T

    Config().set_contraction(True)
    out = operations.apply_operation_matrix(
        states, targets, rho, op, meta=plan, use_contraction=False
    )
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out, expected)
