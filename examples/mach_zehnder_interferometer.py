"""
An example, showing Mach-Zender Interferometer action with PhotonWeave
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import matplotlib.pyplot as plt

from photon_weave.operation import (
    CompositeOperationType,
    FockOperationType,
    Operation,
)
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope


def _probability_one(state: jnp.ndarray) -> float:
    """
    Probability of measuring |1> from either a state vector or density matrix.
    """
    if state.ndim == 2 and state.shape[1] == 1:
        return float(jnp.abs(state[1, 0]) ** 2)
    if state.ndim == 2:
        return float(jnp.real(state[1, 1]))
    if state.ndim == 1:
        return float(jnp.real(state[1]))
    raise ValueError(f"Unexpected state shape {state.shape}")


def mach_zender_probabilities(phase_shift: float):
    # Create one envelope
    env1 = Envelope()
    # Create one photon
    env1.fock.state = 1
    env1.fock.dimensions = 2

    # Other port will consume vacuum
    env2 = Envelope()
    env2.fock.dimensions = 2

    # Generate operators
    bs1 = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)
    ps = Operation(FockOperationType.PhaseShift, phi=phase_shift)
    bs2 = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)

    ce = CompositeEnvelope(env1, env2)
    ce.apply_operation(bs1, env1.fock, env2.fock)
    env1.fock.apply_operation(ps)
    ce.apply_operation(bs2, env1.fock, env2.fock)

    # Extract deterministic probabilities from the reduced states
    prob1 = _probability_one(jnp.asarray(env1.fock.trace_out()))
    prob2 = _probability_one(jnp.asarray(env2.fock.trace_out()))
    return [prob1, prob2]


if __name__ == "__main__":
    angles = jnp.linspace(0, 2 * jnp.pi, 100)
    results = {float(angle): mach_zender_probabilities(angle) for angle in angles}

    measurements_1 = [vals[0] for vals in results.values()]
    measurements_2 = [vals[1] for vals in results.values()]

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(angles, measurements_1, label="Output Port 1 Probability")
    plt.plot(angles, measurements_2, label="Output Port 2 Probability")
    plt.xlabel("Phase Shift (radians)")
    plt.ylabel("Probability")
    plt.title("Mach-Zehnder Interferometer Output Probabilities vs Phase Shift")
    plt.legend()
    plt.grid(True)
    plt.show()
