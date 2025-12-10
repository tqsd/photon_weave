"""
Quantum feature classifier
--------------------------
This small example shows how PhotonWeave can host a quantum-inspired
feature map that is trained with JAX gradients. The circuit applies
parameterized displacements and phase shifts to a single-mode Fock
state and then reads out the photon-number expectation to classify
synthetic data.
"""
from __future__ import annotations

import os

#os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from photon_weave.operation import FockOperationType, Operation
from photon_weave.photon_weave import Config
from photon_weave.state.envelope import Envelope

SPACE_DIMENSIONS = 14


def photon_number_expectation(params: jnp.ndarray, feature: float) -> jnp.ndarray:
    """
    Prepare a small circuit, then use the library's measurement pipeline
    to obtain the photon-number expectation of the Fock mode.
    """
    displacement = params[0] + feature * params[1]
    phase = params[2] + feature * params[3]
    env = Envelope()
    env.fock.dimensions = SPACE_DIMENSIONS
    env.fock.state = 0
    env.apply_operation(
        Operation(FockOperationType.Displace, alpha=displacement),
        env.fock,
    )
    env.apply_operation(
        Operation(FockOperationType.PhaseShift, phi=phase),
        env.fock,
    )
    probabilities, _ = env.fock.measure_expectation()
    levels = jnp.arange(probabilities.shape[0], dtype=probabilities.dtype)
    return jnp.dot(levels, probabilities)


def predict(params: jnp.ndarray, feature: float) -> jnp.ndarray:
    """Map the photon-number expectation to [0, 1] with a logistic layer."""
    expectation = photon_number_expectation(params, feature)
    bias = params[4]
    return jax.nn.sigmoid(expectation + bias)


def loss(params, features, labels):
    """Mean squared error between the model and the synthetic labels."""
    batched = jax.vmap(partial(predict, params))
    predictions = batched(features)
    return jnp.mean((predictions - labels) ** 2)


loss_and_grad = jax.value_and_grad(loss)


def make_trainer(
    features: jnp.ndarray,
    labels: jnp.ndarray,
    steps: int = 100,
    lr: float = 0.25,
):
    """
    Create a jitted training function with dataset closed over (static).
    Features and labels are treated as compile-time constants inside the jit.
    """
    features = jnp.asarray(features)
    labels = jnp.asarray(labels)

    @partial(jax.jit, static_argnames=("steps", "lr"))
    def train(init_params: jnp.ndarray, *, steps: int = steps, lr: float = lr):
        def body(params, _):
            current_loss, grads = loss_and_grad(params, features, labels)
            new_params = params - lr * grads
            return new_params, current_loss

        return jax.lax.scan(body, init_params, jnp.arange(steps))

    return train


def build_dataset() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Binary class labels on a 1D input for illustration."""
    features = jnp.linspace(-1.0, 1.0, 25)
    labels = (features > 0).astype(jnp.float32)
    return features, labels


if __name__ == "__main__":
    conf = Config()
    conf.set_contraction(True)
    conf.set_dynamic_dimensions(False)
    steps = 100
    features, labels = build_dataset()
    init_params = jnp.array([0.5, 0.2, 0.1, 0.3, -0.5])
    train = make_trainer(features, labels, steps=steps, lr=0.25)
    trained, losses = train(init_params)
    losses = jnp.asarray(losses)
    for step in range(0, steps, 10):
        print(f"step {step:>2d} loss {float(losses[step]):.4f}")

    predictions = jax.vmap(partial(predict, trained))(features)
    plt.plot(features, labels, "o", label="target")
    plt.plot(features, predictions, label="qml prediction")
    plt.xlabel("Feature")
    plt.ylabel("Probability")
    plt.title("Photon-number-based quantum feature classifier")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
