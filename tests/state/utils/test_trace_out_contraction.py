import jax.numpy as jnp

from photon_weave.photon_weave import Config
from photon_weave.state.utils.trace_out import trace_out_matrix
from photon_weave.state.utils.shape_planning import build_plan


class DummyState:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions


def _basis_state(index: int, dim: int) -> jnp.ndarray:
    vec = jnp.zeros((dim, 1), dtype=jnp.complex128)
    return vec.at[index, 0].set(1.0)


def test_trace_out_contraction_noncontiguous_matches_full_operator():
    """
    Ensure contraction-based trace-out hits the cached einsum path and matches
    the explicit reorder+trace for noncontiguous targets.
    """
    states = [DummyState(2), DummyState(3), DummyState(2)]
    # Keep subsystems 0 and 2 (trace out middle)
    targets = [states[0], states[2]]
    plan = build_plan(states, targets)

    # |0,1,0> basis state -> density matrix
    state = _basis_state(2, 12)  # index 2 corresponds to |0,1,0> in ordering
    rho = state @ jnp.conj(state).T

    # Opt-einsum contraction path
    Config().set_contraction(True)
    out_contract = trace_out_matrix(
        states, targets, rho, meta=plan.meta, use_contraction=True
    )

    # Explicit reorder + full trace using kron
    rho_tensor = rho.reshape((2, 3, 2, 2, 3, 2))
    # Move targets (0 and 2) to front, trace out middle (index 1)
    rho_reordered = jnp.transpose(rho_tensor, (0, 2, 1, 3, 5, 4))
    traced = jnp.trace(rho_reordered, axis1=2, axis2=5)
    expected = traced.reshape((4, 4))

    assert jnp.allclose(out_contract, expected)
