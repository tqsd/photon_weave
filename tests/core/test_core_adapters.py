import jax
import jax.numpy as jnp
import pytest

from photon_weave._math.ops import x_operator
from photon_weave.core import adapters
from photon_weave.photon_weave import Config


def _basis_state(index: int, dim: int) -> jnp.ndarray:
    v = jnp.zeros((dim, 1), dtype=jnp.complex128)
    v = v.at[index, 0].set(1)
    return v


def _density(v: jnp.ndarray) -> jnp.ndarray:
    return v @ jnp.conj(v.T)


@pytest.fixture(autouse=True)
def restore_contraction_flag():
    cfg = Config()
    prev = cfg.contractions
    yield
    cfg.set_contraction(prev)


@pytest.mark.parametrize("use_contraction", [False, True])
def test_apply_operation_vector_matches_full_kron(use_contraction):
    Config().set_contraction(use_contraction)
    dims = (2, 2)
    op = x_operator()
    state = _basis_state(0, 4)  # |00>
    out = adapters.apply_operation_vector(
        dims, (0,), state, op, use_contraction=use_contraction
    )
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ state
    assert jnp.allclose(out, expected)


@pytest.mark.parametrize("use_contraction", [False, True])
def test_apply_operation_matrix_matches_full_kron(use_contraction):
    Config().set_contraction(use_contraction)
    dims = (2, 2)
    op = x_operator()
    state_vec = _basis_state(0, 4)
    rho = _density(state_vec)
    out = adapters.apply_operation_matrix(
        dims, (0,), rho, op, use_contraction=use_contraction
    )
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out, expected)


def test_apply_kraus_matrix_single_op_matches_unitary():
    dims = (2, 2)
    op = x_operator()
    state_vec = _basis_state(0, 4)
    rho = _density(state_vec)
    kraus = jnp.stack([op])
    out = adapters.apply_kraus_matrix(dims, (0,), rho, kraus)
    full_op = jnp.kron(op, jnp.eye(2))
    expected = full_op @ rho @ jnp.conj(full_op.T)
    assert jnp.allclose(out, expected)


def test_measure_vector_collapses_and_decodes_outcomes():
    dims = (2, 2)
    state = _basis_state(0, 4)  # |00>
    key = jax.random.PRNGKey(0)
    outcome, post, _ = adapters.measure_vector(dims, (0,), state, key)
    assert outcome == 0
    assert post.shape == (2, 1)
    assert jnp.allclose(post, _basis_state(0, 2))


def test_measure_matrix_collapses_and_decodes_outcomes():
    dims = (2, 2)
    rho = _density(_basis_state(0, 4))  # |00><00|
    key = jax.random.PRNGKey(0)
    outcome, post, _ = adapters.measure_matrix(dims, (0,), rho, key)
    assert outcome == 0
    assert post.shape == (2, 2)
    assert jnp.allclose(post, jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128))


def test_measure_povm_matrix_projective():
    dims = (2, 2)
    rho = _density(_basis_state(0, 4))
    proj0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128)
    proj1 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128)
    operators = jnp.stack([proj0, proj1])
    key = jax.random.PRNGKey(0)
    outcome, post, _ = adapters.measure_povm_matrix(dims, (0,), operators, rho, key)
    assert outcome == 0
    assert jnp.allclose(post, rho)


def test_trace_out_matrix_reduces_state():
    dims = (2, 2)
    rho = _density(_basis_state(0, 4))
    reduced = adapters.trace_out_matrix(dims, (0,), rho)
    assert reduced.shape == (2, 2)
    assert jnp.allclose(reduced, jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128))


def test_measure_vector_with_probs_returns_probs_and_post():
    dims = (2, 2)
    state = _basis_state(1, 4)  # |01>
    key = jax.random.PRNGKey(0)
    outcome, post, probs, _ = adapters.measure_vector_with_probs(dims, (0,), state, key)
    assert outcome == 0
    assert jnp.allclose(probs, jnp.array([1.0, 0.0]))
    assert post.shape == (2, 1)
    assert jnp.allclose(post, _basis_state(1, 2))


def test_measure_matrix_with_probs_returns_probs_and_post():
    dims = (2, 2)
    rho = _density(_basis_state(1, 4))  # |01><01|
    key = jax.random.PRNGKey(0)
    outcome, post, probs, _ = adapters.measure_matrix_with_probs(dims, (0,), rho, key)
    assert outcome == 0
    assert jnp.allclose(probs, jnp.array([1.0, 0.0]))
    assert post.shape == (2, 2)
    assert jnp.allclose(post, jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128))


def test_measure_vector_expectation_matches_reduced_density():
    dims = (2, 2)
    state = jnp.zeros((4, 1), dtype=jnp.complex128)
    amp = 1 / jnp.sqrt(2)
    state = state.at[0, 0].set(amp)  # |00>
    state = state.at[2, 0].set(amp)  # |10>
    probs, post = adapters.measure_vector_expectation(dims, (0,), state)
    assert jnp.allclose(probs, jnp.array([0.5, 0.5]))
    assert post.shape == (2, 2)
    assert jnp.allclose(post, jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128))


def test_measure_matrix_expectation_matches_trace_out():
    dims = (2, 2)
    rho = _density(_basis_state(0, 4))  # |00><00|
    probs, post = adapters.measure_matrix_expectation(dims, (0,), rho)
    assert jnp.allclose(probs, jnp.array([1.0, 0.0]))
    assert post.shape == (2, 2)
    assert jnp.allclose(post, jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128))
