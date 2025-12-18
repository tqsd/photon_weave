import unittest
import jax
import jax.numpy as jnp
import pytest

from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.operation import Operation, CompositeOperationType
from photon_weave.photon_weave import Config
from photon_weave.state.utils.measurements import (
    measure_vector,
    measure_matrix,
    pnr_transition_matrix,
    pnr_povm,
    gaussian_jitter_kernel,
    measure_matrix_jit,
    measure_vector_jit,
    measure_matrix_jit_with_probs,
    measure_vector_expectation,
    measure_matrix_expectation,
    measure_vector_jit_with_probs,
    measure_pnr_vector,
    measure_pnr_matrix,
)
from photon_weave.state.utils.shape_planning import build_plan, build_meta


class DummyState:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions


def _basis_state(index: int, dim: int) -> jnp.ndarray:
    v = jnp.zeros((dim, 1), dtype=jnp.complex128)
    return v.at[index, 0].set(1.0)


def _density(v: jnp.ndarray) -> jnp.ndarray:
    return v @ jnp.conj(v.T)


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


class TestMeasurementUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        self._prev_contractions = self.cfg.contractions
        self._prev_dynamic = self.cfg.dynamic_dimensions
        self._prev_jit = self.cfg.use_jit

    def tearDown(self) -> None:
        self.cfg.set_contraction(self._prev_contractions)
        self.cfg.set_dynamic_dimensions(self._prev_dynamic)
        self.cfg.set_use_jit(self._prev_jit)

    def test_measurement_vector(self) -> None:
        env1 = Envelope()
        env1.fock.state = 1
        env2 = Envelope()
        env3 = Envelope()
        env4 = Envelope()

        measurement_probabilities = {}

        def log_probabilities(fock, probabilities):
            measurement_probabilities[fock] = probabilities

        # create the state
        op = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)

        ce = CompositeEnvelope(env1, env2, env3, env4)
        ce.apply_operation(op, env1.fock, env2.fock)
        ce.apply_operation(op, env1.fock, env3.fock)
        ce.apply_operation(op, env4.fock, env2.fock)

        ce.reorder(env1.fock, env2.fock, env3.fock, env4.fock)

        for env in [env1, env2, env3, env4]:
            measure_vector(
                [env1.fock, env2.fock, env3.fock, env4.fock],
                [env.fock],
                ce.states[0].state,
                prob_callback=log_probabilities,
            )
        for state, item in measurement_probabilities.items():
            assert jnp.isclose(float(item[0]), 0.75, atol=1e-5)
            assert jnp.isclose(float(item[1]), 0.25, atol=1e-5)

    def test_measurement_matrix(self) -> None:
        env1 = Envelope()
        env1.fock.state = 1
        env2 = Envelope()
        env3 = Envelope()
        env4 = Envelope()

        measurement_probabilities = {}

        def log_probabilities(fock, probabilities):
            measurement_probabilities[fock] = probabilities

        # create the state
        op = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)

        ce = CompositeEnvelope(env1, env2, env3, env4)
        ce.apply_operation(op, env1.fock, env2.fock)
        ce.apply_operation(op, env1.fock, env3.fock)
        ce.apply_operation(op, env4.fock, env2.fock)

        ce.reorder(env1.fock, env2.fock, env3.fock, env4.fock)

        env1.fock.expand()
        print(ce.states[0].state)

        for env in [env1, env2, env3, env4]:
            measure_matrix(
                [env1.fock, env2.fock, env3.fock, env4.fock],
                [env.fock],
                ce.states[0].state,
                prob_callback=log_probabilities,
            )
        for state, item in measurement_probabilities.items():
            print(item)
            assert jnp.isclose(float(item[0]), 0.75, atol=1e-5)
            assert jnp.isclose(float(item[1]), 0.25, atol=1e-5)

    def test_pnr_transition_and_povm(self) -> None:
        eta = 0.6
        dark = 50.0
        window = 1e-9
        P = pnr_transition_matrix(max_photons=3, eta=eta, dark_rate=dark, window=window)
        col_sums = P.sum(axis=0)
        self.assertTrue(jnp.allclose(col_sums, jnp.ones_like(col_sums), atol=1e-6))
        povm = pnr_povm(3, eta=eta, dark_rate=dark, window=window)
        completeness = povm.sum(axis=0)
        self.assertTrue(jnp.allclose(completeness, jnp.eye(4), atol=1e-6))

    def test_gaussian_jitter_kernel_normalizes(self) -> None:
        J = gaussian_jitter_kernel(num_bins=4, bin_width=1.0, jitter_std=0.2)
        col_sums = J.sum(axis=0)
        self.assertTrue(jnp.allclose(col_sums, jnp.ones_like(col_sums), atol=1e-6))


def test_measure_vector_jit_accepts_shape_plan_and_decodes_outcome():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    state = _basis_state(0, 4)  # |00>
    key = jax.random.PRNGKey(0)

    outcomes, post, _ = measure_vector_jit(states, targets, state, key, meta=plan)
    assert outcomes[targets[0]] == 0
    assert post.shape == (2, 1)
    assert jnp.allclose(post, _basis_state(0, 2))


def test_measure_matrix_jit_accepts_shape_plan_and_decodes_outcome():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    state = _basis_state(0, 4)  # |00>
    rho = _density(state)
    key = jax.random.PRNGKey(0)

    outcomes, post, _ = measure_matrix_jit(states, targets, rho, key, meta=plan)
    assert outcomes[targets[0]] == 0
    assert post.shape == (2, 2)
    assert jnp.allclose(post, jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128))


def test_measure_vector_jit_accepts_dims_meta_with_contraction_flag_restored():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    meta = build_meta(states, targets)
    state = _basis_state(0, 4)  # |00>
    key = jax.random.PRNGKey(0)

    Config().set_contraction(False)
    outcomes, post, _ = measure_vector_jit(states, targets, state, key, meta=meta)
    assert outcomes[targets[0]] == 0
    assert post.shape == (2, 1)
    assert jnp.allclose(post, _basis_state(0, 2))


def test_measure_vector_jit_with_probs_accepts_shape_plan():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    state = _basis_state(0, 4)  # |00>
    key = jax.random.PRNGKey(0)

    outcomes, post, probs, _ = measure_vector_jit_with_probs(
        states, targets, state, key, meta=plan
    )
    assert outcomes[targets[0]] == 0
    assert jnp.allclose(probs, jnp.array([1.0, 0.0]))
    assert post.shape == (2, 1)
    assert jnp.allclose(post, _basis_state(0, 2))


def test_measure_matrix_jit_with_probs_accepts_shape_plan():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    rho = _density(_basis_state(0, 4))
    key = jax.random.PRNGKey(0)

    outcomes, post, probs, _ = measure_matrix_jit_with_probs(
        states, targets, rho, key, meta=plan
    )
    assert outcomes[targets[0]] == 0
    assert jnp.allclose(probs, jnp.array([1.0, 0.0]))
    assert post.shape == (2, 2)
    assert jnp.allclose(post, jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128))


def test_measure_vector_expectation_accepts_shape_plan():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    state = jnp.zeros((4, 1), dtype=jnp.complex128)
    state = state.at[0, 0].set(1.0)
    state = state.at[2, 0].set(1.0)
    state = state / jnp.sqrt(2)

    probs, post = measure_vector_expectation(states, targets, state, meta=plan)
    assert jnp.allclose(probs, jnp.array([0.5, 0.5]))
    assert post.shape == (2, 2)
    assert jnp.allclose(post, jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128))


def test_measure_matrix_expectation_accepts_dims_meta_with_jit_flag():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    meta = build_meta(states, targets)
    state = _basis_state(0, 4)
    rho = _density(state)

    Config().set_use_jit(True)
    probs, post = measure_matrix_expectation(states, targets, rho, meta=meta)
    assert jnp.allclose(probs, jnp.array([1.0, 0.0]))
    assert post.shape == (2, 2)
    assert jnp.allclose(post, jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128))


def test_measure_pnr_vector_requires_key_for_jit_path():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    state = _basis_state(0, 4)
    with pytest.raises(ValueError):
        measure_pnr_vector(
            states,
            targets,
            state,
            meta=plan,
            efficiency=1.0,
            dark_rate=0.0,
            detection_window=1.0,
            jitter_std=0.0,
            key=None,
        )


def test_measure_pnr_vector_jit_path_uses_given_key():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    state = _basis_state(0, 4)
    key = jax.random.PRNGKey(0)
    outcomes, _, jitter, next_key = measure_pnr_vector(
        states,
        targets,
        state,
        meta=plan,
        efficiency=1.0,
        dark_rate=0.0,
        detection_window=1.0,
        jitter_std=0.0,
        key=key,
    )
    assert outcomes[targets[0]] == 0
    assert jitter.shape == (1,)
    assert isinstance(next_key, jnp.ndarray)


def test_measure_pnr_matrix_requires_key_for_jit_path():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    rho = _density(_basis_state(0, 4))
    with pytest.raises(ValueError):
        measure_pnr_matrix(
            states,
            targets,
            rho,
            meta=plan,
            efficiency=1.0,
            dark_rate=0.0,
            detection_window=1.0,
            jitter_std=0.0,
            key=None,
        )


def test_measure_pnr_matrix_jit_path_uses_given_key():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)
    rho = _density(_basis_state(0, 4))
    key = jax.random.PRNGKey(0)
    outcomes, _, jitter, next_key = measure_pnr_matrix(
        states,
        targets,
        rho,
        meta=plan,
        efficiency=1.0,
        dark_rate=0.0,
        detection_window=1.0,
        jitter_std=0.0,
        key=key,
    )
    assert outcomes[targets[0]] == 0
    assert jitter.shape == (1,)
    assert isinstance(next_key, jnp.ndarray)


def test_measure_vector_expectation_grad():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)

    def prob_sum(theta):
        state = jnp.zeros((4, 1), dtype=jnp.complex128)
        state = state.at[0, 0].set(jnp.cos(theta))
        state = state.at[2, 0].set(jnp.sin(theta))
        state = state / jnp.linalg.norm(state)
        probs, _ = measure_vector_expectation(states, targets, state, meta=plan)
        return probs[0]

    grad_fn = jax.grad(prob_sum)
    g = grad_fn(0.5)
    assert jnp.isfinite(g)


def test_measure_matrix_expectation_grad():
    states = [DummyState(2), DummyState(2)]
    targets = [states[0]]
    plan = build_plan(states, targets)

    def prob_sum(theta):
        state = jnp.zeros((4, 1), dtype=jnp.complex128)
        state = state.at[0, 0].set(jnp.cos(theta))
        state = state.at[2, 0].set(jnp.sin(theta))
        state = state / jnp.linalg.norm(state)
        rho = _density(state)
        probs, _ = measure_matrix_expectation(states, targets, rho, meta=plan)
        return probs[0]

    grad_fn = jax.grad(prob_sum)
    g = grad_fn(0.5)
    assert jnp.isfinite(g)
