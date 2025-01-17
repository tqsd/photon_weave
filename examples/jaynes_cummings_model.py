"""
Simple Jaynes-Cummings Model example
------------------------------------
Here we demonstrate how complex a Hamiltonian can be
turned into an operator using PhotonWeave Expressions
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.custom_state import CustomState
from photon_weave.operation import Operation, CompositeOperationType, FockOperationType
from photon_weave.photon_weave import Config
from photon_weave.extra.expression_interpreter import interpreter

# Create an envelope with one photon |1>
env = Envelope()
env.fock.state = 1
env.fock.dimensions = 2
# Create the matter system
qubit = CustomState(2)
qubit.state = 0

# We study the systems together
ce = CompositeEnvelope(env, qubit)

# Physical constants
h_bar = 1.054571817e-34  # Reduced Planck's constant (Joule * seconds)
h_bar = 1

# System parameters
w_field = 1 * jnp.pi  # Angular frequency of the field (radians per second, e.g., 5 GHz)
w_qubit = 1 * jnp.pi  # Angular frequency of the qubit (radians per second, e.g., 5 GHz)
g = (
    1 * jnp.pi
)  # Coupling strength between the qubit and field (radians per second, e.g., 100 MHz)

# Time parameters
t_delta = 0.01  # Time step for simulation (seconds, e.g., 1 nanosecond)
t_max = 3  # Total simulation time (seconds, e.g., 1 microsecond)

# Define the relevant qubit operators
sigma_z = jnp.array([[1, 0], [0, -1]])
sigma_plus = jnp.array([[0, 1], [0, 0]])
sigma_minus = jnp.array([[0, 0], [1, 0]])


# Define the relevant Fock operators
def annihilation_operator(n):
    data = jnp.sqrt(jnp.arange(1, n))
    return jnp.diag(data, k=1)


def creation_operator(n):
    data = jnp.sqrt(jnp.arange(1, n))
    return jnp.diag(data, k=-1)


# Define the expression context
context = {
    "a": lambda dims: annihilation_operator(dims[0]),
    "a_dag": lambda dims: creation_operator(dims[0]),
    "i_p": lambda dims: jnp.eye(dims[0]),
    "s_z": lambda dims: sigma_z,
    "s_plus": lambda dims: sigma_plus,
    "s_minus": lambda dims: sigma_minus,
    "s_i": lambda dims: jnp.eye(2),
}

# Field Hamiltonian
H_field = (
    "s_mult",
    h_bar,
    w_field,
    (
        "kron",
        (
            "m_mult",
            "a_dag",
            "a",
        ),
        "s_i",
    ),
)

# Qubit Hamiltonian
H_qubit = ("s_mult", h_bar, w_qubit / 2, ("kron", "i_p", "s_z"))

# Interaction Hamiltonian
H_interaction = (
    "s_mult",
    h_bar,
    g,
    ("add", ("kron", "a", "s_plus"), ("kron", "a_dag", "s_minus")),
)

# Unitary evolution Hamiltonian
expr = (
    "expm",
    ("s_mult", -1j, ("add", H_field, H_qubit, H_interaction), t_delta, 1 / h_bar),
)

ce.combine(env.fock, qubit)
interractions = jnp.arange(0, t_max, t_delta)

jc_interraction = Operation(
    CompositeOperationType.Expression,
    expr=expr,
    context=context,
    state_types=[Fock, CustomState],
)
C = Config()
C.set_contraction(False)
qubit_excited_populations = []
for i, step in enumerate(interractions):
    ce.apply_operation(jc_interraction, env.fock, qubit)
    combined_state = ce.states[0].state
    qubit_reduced = qubit.trace_out()
    qubit_reduced = qubit_reduced / jnp.linalg.norm(qubit_reduced)
    print(qubit_reduced)
    qubit_excited_populations.append(jnp.abs(qubit_reduced[1][0]) ** 2)

# plt.figure(figsize=(6, 3.375))
# plt.plot(interractions, qubit_excited_populations, label="Qubit Excited State Population")
# plt.xlabel("Time (s)")
# plt.ylabel("Population")
# plt.title("Rabi Oscillations")
# plt.legend()
# plt.grid()
# #plt.show()
# plt.savefig("plots/jc.png", dpi=1000, bbox_inches="tight")  # Save as a high-quality PNG
