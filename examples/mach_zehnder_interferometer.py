"""
An example, showing Mach-Zender Interferometer action with PhotonWeave
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from photon_weave.operation import CompositeOperationType, FockOperationType, Operation
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope


def mach_zender_single_shot(phase_shift: float):
    # Create one envelope
    env1 = Envelope()
    # Create one photon
    env1.fock.state = 1

    # Other port will consume vacuum
    env2 = Envelope()

    # Generate operators
    bs1 = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)
    ps = Operation(FockOperationType.PhaseShift, phi=phase_shift)
    bs2 = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)

    ce = CompositeEnvelope(env1, env2)
    ce.apply_operation(bs1, env1.fock, env2.fock)
    env1.fock.apply_operation(ps)
    ce.apply_operation(bs2, env1.fock, env2.fock)

    out1 = env1.fock.measure()
    out2 = env2.fock.measure()
    return [out1[env1.fock], out2[env2.fock]]


if __name__ == "__main__":
    num_shots = 1000
    angles = jnp.linspace(0, 2 * jnp.pi, 25)
    results = {float(angle): [] for angle in angles}
    for angle in angles:
        for _ in range(num_shots):
            shot_result = mach_zender_single_shot(angle)
            results[float(angle)].append(shot_result)

    measurements_1 = []
    measurements_2 = []
    for angle, shots in results.items():
        counts_1 = sum(shot[0] for shot in shots) / num_shots
        counts_2 = sum(shot[1] for shot in shots) / num_shots
        measurements_1.append(counts_1)
        measurements_2.append(counts_2)

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
