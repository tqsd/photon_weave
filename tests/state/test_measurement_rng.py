import jax
import jax.numpy as jnp

from photon_weave.photon_weave import Config
from photon_weave.state.envelope import Envelope
from photon_weave.state.expansion_levels import ExpansionLevel


def _bell_like_envelope():
    env = Envelope()
    # Simple entangled-like superposition over fock/polarization for determinism check
    env.fock.dimensions = 2
    env.fock.state = jnp.array([[1.0], [1.0]]) / jnp.sqrt(2)
    env.fock.expansion_level = ExpansionLevel.Vector
    env.polarization.state = jnp.array([[1.0], [0.0]])
    env.polarization.expansion_level = ExpansionLevel.Vector
    env.combine()
    return env


def test_measure_with_same_key_is_reproducible_non_jit():
    Config().set_use_jit(False)
    key = jax.random.PRNGKey(123)
    env = _bell_like_envelope()
    out1 = env.measure(key=key)
    env = _bell_like_envelope()
    out2 = env.measure(key=key)
    assert sorted(out1.values()) == sorted(out2.values())


def test_measure_with_different_keys_can_differ_non_jit():
    Config().set_use_jit(False)
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)
    env = _bell_like_envelope()
    out1 = env.measure(key=key1)
    env = _bell_like_envelope()
    out2 = env.measure(key=key2)
    assert sorted(out1.values()) != sorted(out2.values())
