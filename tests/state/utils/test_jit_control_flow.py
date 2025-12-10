import jax
import jax.numpy as jnp
from jax import lax

from photon_weave.core import adapters
from photon_weave.operation import (
    CompositeOperationType,
    FockOperationType,
    Operation,
)
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.utils.measurements import measure_matrix_jit
from photon_weave.state.utils.operations import (
    apply_operation_matrix,
    apply_operation_vector,
)
from photon_weave.state.utils.shape_planning import (
    build_plan,
    compiled_kernels,
)


def _basis_vector(dim: int, idx: int = 0) -> jnp.ndarray:
    vec = jnp.zeros((dim, 1), dtype=jnp.complex128)
    return vec.at[idx, 0].set(1.0)


def _density_from_vector(vec: jnp.ndarray) -> jnp.ndarray:
    return vec @ jnp.conj(vec.T)


def test_envelope_apply_in_fori_loop_with_plan_jitted():
    env = Envelope()
    env.fock.dimensions = 2
    state_objs = [env.fock]
    targets = [env.fock]
    plan = build_plan(state_objs, targets)
    op = Operation(FockOperationType.PhaseShift, phi=0.3)
    op.dimensions = [2]
    psi0 = _basis_vector(2, 0)

    def body(_, acc):
        return apply_operation_vector(
            state_objs,
            targets,
            acc,
            op.operator,
            meta=plan,
            use_contraction=False,
        )

    @jax.jit
    def run():
        return lax.fori_loop(0, 4, body, psi0)

    out = run()
    assert out.shape == psi0.shape


def test_composite_measure_in_conditional_jitted():
    env1 = Envelope()
    env2 = Envelope()
    env1.fock.dimensions = 2
    env2.fock.dimensions = 2
    env1.fock.state = 1
    env2.fock.state = 0
    ce = CompositeEnvelope(env1, env2)
    ce.combine(env1.fock, env2.fock)
    ps = ce.product_states[0]
    state_objs = ps.state_objs
    # Measure all targets so post-state shape matches rho for cond branches
    targets = state_objs
    plan = build_plan(state_objs, targets)
    rho_full = _density_from_vector(ps.state)
    rho_scalar = jnp.reshape(rho_full[0, 0], (1, 1))
    key = jax.random.PRNGKey(0)

    @jax.jit
    def maybe_measure(do_measure: bool):
        def _measure(_):
            _, post, _ = adapters.measure_matrix_jit_meta(plan.meta, rho_full, key)
            return post

        return lax.cond(do_measure, _measure, lambda _: rho_scalar, operand=None)

    measured = maybe_measure(True)
    skipped = maybe_measure(False)
    assert measured.shape == rho_scalar.shape
    assert skipped.shape == rho_scalar.shape


def test_composite_apply_in_while_loop_with_compiled_kernels():
    env1 = Envelope()
    env2 = Envelope()
    env1.fock.dimensions = 2
    env2.fock.dimensions = 2
    env1.fock.state = 1
    env2.fock.state = 0
    ce = CompositeEnvelope(env1, env2)
    ce.combine(env1.fock, env2.fock)
    ps = ce.product_states[0]
    plan = build_plan(ps.state_objs, (env1.fock, env2.fock))
    kernels = compiled_kernels(plan)
    bs = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)
    bs.dimensions = [2, 2]
    rho = _density_from_vector(ps.state)

    def cond(carry):
        i, _ = carry
        return i < 3

    def body(carry):
        i, mat = carry
        updated = kernels.apply_op_matrix(mat, bs.operator, use_contraction=True)
        return i + 1, updated

    @jax.jit
    def run():
        _, out = lax.while_loop(cond, body, (0, rho))
        return out

    result = run()
    assert result.shape == rho.shape


def test_beamsplitter_probabilities_jitted_expectation():
    """
    Physics sanity check: a 50/50 beamsplitter on |1,0> should yield
    equal probabilities for |1,0> and |0,1>.
    """
    env1 = Envelope()
    env2 = Envelope()
    env1.fock.dimensions = 2
    env2.fock.dimensions = 2
    env1.fock.state = 1
    env2.fock.state = 0
    ce = CompositeEnvelope(env1, env2)
    ce.combine(env1.fock, env2.fock)
    ps = ce.product_states[0]
    plan = build_plan(ps.state_objs, (env1.fock, env2.fock))
    kernels = compiled_kernels(plan)

    bs = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)
    bs.dimensions = [2, 2]

    @jax.jit
    def run():
        # Work with the pure state vector to keep amplitudes explicit.
        psi = kernels.apply_op_vector(ps.state, bs.operator, use_contraction=True)
        rho = _density_from_vector(psi)
        # Basis ordering is |00>, |01>, |10>, |11>; pick the single-photon slots.
        return jnp.real(jnp.diag(rho))

    probs = run()
    assert jnp.allclose(probs[1], 0.5, atol=1e-2)
    assert jnp.allclose(probs[2], 0.5, atol=1e-2)
    assert jnp.allclose(probs.sum(), 1.0, atol=1e-3)
