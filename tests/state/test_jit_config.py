import jax.numpy as jnp

from photon_weave.photon_weave import Config
from photon_weave.state.envelope import Envelope
from photon_weave.state.composite_envelope import CompositeEnvelope


def test_envelope_expand_respects_use_jit(monkeypatch):
    env = Envelope()
    env.fock.state = 1
    env.fock.dimensions = 2
    env.combine()

    # Force use_jit True
    Config().set_use_jit(True)

    # Monkeypatch state_expand_jit to observe it was called
    calls = {}

    def fake_expand(state, level, dims):
        calls["called"] = True
        return jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128), level

    monkeypatch.setattr(
        "photon_weave.state.envelope.state_expand_jit", fake_expand
    )
    env.expand()
    assert calls.get("called", False)

    # Restore default
    Config().set_use_jit(False)


def test_product_state_expand_respects_use_jit(monkeypatch):
    env1 = Envelope()
    env1.fock.state = 1
    env1.fock.dimensions = 2
    env2 = Envelope()
    env2.polarization.state = env2.polarization.state  # keep default

    ce = CompositeEnvelope(env1, env2)
    ce.combine(env1.fock, env2.polarization)
    ps = ce.product_states[0]

    Config().set_use_jit(True)
    calls = {}

    def fake_expand(state, level, dims):
        calls["called"] = True
        return jnp.eye(dims, dtype=jnp.complex128), level

    monkeypatch.setattr(
        "photon_weave.state.composite_envelope.state_expand_jit", fake_expand
    )
    ps.expand()
    assert calls.get("called", False)

    Config().set_use_jit(False)
