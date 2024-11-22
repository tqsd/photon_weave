"""
An example, showing Mach-Zender Interferometer action with PhotonWeave
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from photon_weave.operation import CompositeOperationType, FockOperationType, Operation
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope


def mach_zender_single_shot(phase_shift: float) -> list[int]:
    """Return photon count in each port of a Mach-Zehnder Interferometer.

    Parameters
    ----------
    phase_shift : float
        Phase shift between the two arms of the interferometer.

    Returns
    -------
    list[int]
        Photon count in each port of the interferometer.
    """
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
    from tqdm import tqdm
    import numpy as np
    num_shots = 100
    angles = jnp.linspace(0, 2 * jnp.pi, 10)
    # (num_angles, num_shots , num_ports) Pre allocating this yields a x2 speedup for free
    results = np.zeros((angles.shape[0], num_shots, 2))
    pbar = tqdm(total=len(angles) * num_shots, desc="Simulating Mach-Zehnder Interferometer")
    for i, angle in enumerate(angles):
        for j in range(num_shots):
            shot_result = mach_zender_single_shot(angle)
            # results[float(angle)].append(shot_result)
            results[i, j, :] = shot_result
            pbar.update(1)
    pbar.close()
    measurements_1 = results.mean(axis=1)[:, 0]
    measurements_2 = results.mean(axis=1)[:, 1]

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
