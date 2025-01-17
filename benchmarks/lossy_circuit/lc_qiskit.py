import numpy as np
from qiskit.quantum_info import Operator, Statevector
from qiskit.quantum_info.operators import Operator
from scipy.linalg import expm
import argparse
import json


class ArrayMemoryUsageTracker:
    def __init__(self):
        self.operator_sizes = []
        self.state_sizes = []

    def record_state_size(self, *states):
        mem = 0
        for s in states:
            mem += s.nbytes
        self.state_sizes.append(mem)

    def record_operator_size(self, *operators):
        mem = 0
        for op in operators:
            mem += op.nbytes
        self.operator_sizes.append(mem)


def fock_beamsplitter(dim, theta=np.pi / 4):
    """
    Create a beamsplitter operator in Fock space for two modes using tensor products.

    Args:
        dim (int): Dimension of the Fock space for each mode.
        theta (float): Beamsplitter angle.

    Returns:
        Operator: Beamsplitter operator as a Qiskit Operator.
    """
    # Create annihilation operators for each mode
    a = np.diag(np.sqrt(np.arange(1, dim)), 1)

    # Tensor product to span both modes
    a1 = np.kron(a, np.eye(dim))
    a2 = np.kron(np.eye(dim), a)

    # Beamsplitter Hamiltonian
    H_bs = 1j * theta * (np.dot(a1.T, a2) + np.dot(a2.T, a1))

    # Exponential to get the unitary operator
    U_bs = expm(H_bs)
    # U_bs = U_bs.reshape([dim, dim, dim, dim])

    return Operator(U_bs, input_dims=(dim, dim), output_dims=(dim, dim))


def conditional_annihilation(dim):
    """
    Creates an operator which acts as identity on the vacuum state,
    but annihilates a photon if present
    """
    # Initialize the operator matrix
    op_matrix = np.zeros((dim, dim), dtype=complex)

    # Populate the matrix
    for n in range(1, dim):  # Start from 1 to skip the vacuum state
        op_matrix[n - 1, n] = n  # Standard annihilation (normalized)

    # Add identity for the vacuum state
    op_matrix[0, 0] = 1  # Leave |0⟩ unchanged

    return Operator(op_matrix)


def lossy_bs_circuit(initial_state, lossy):
    AMUT = ArrayMemoryUsageTracker()

    # Define dimensions for the state
    dim = 4 * (initial_state + 1)

    # Initial state represented as a Statevector (tensor product of basis states)
    env1 = Statevector.from_int(initial_state, dim)
    env2 = Statevector.from_int(initial_state, dim)
    env3 = Statevector.from_int(initial_state, dim)
    env4 = Statevector.from_int(initial_state, dim)

    # Initial memory consumption
    AMUT.record_state_size(env1.data, env2.data, env3.data, env4.data)
    AMUT.record_operator_size()

    # Define beamsplitter operation (simplified matrix)

    BS = Operator(fock_beamsplitter(dim))
    composite_state_1 = env1.expand(env2)
    composite_state_1.evolve(BS)
    if lossy:
        A_cond = conditional_annihilation(dim).tensor(conditional_annihilation(dim))
        composite_state_1.evolve(A_cond)
        AMUT.record_operator_size(BS.data, A_cond.data)
        del A_cond
    else:
        AMUT.record_operator_size(BS.data)
    AMUT.record_state_size(composite_state_1.data, env3.data, env4.data)

    composite_state_2 = env3.expand(env4)
    composite_state_2.evolve(BS)
    if lossy:
        A_cond = conditional_annihilation(dim).tensor(conditional_annihilation(dim))
        composite_state_2.evolve(A_cond)
        AMUT.record_operator_size(BS.data, A_cond.data)
        del A_cond
    else:
        AMUT.record_operator_size(BS.data)
    AMUT.record_state_size(composite_state_1.data, composite_state_2.data)

    composite_state = composite_state_1.expand(composite_state_2)
    BS_0 = Operator(np.eye(dim)).tensor(BS).tensor(Operator(np.eye(dim)))
    composite_state.evolve(BS_0)
    if lossy:
        A_cond = (
            Operator(np.eye(dim))
            .tensor(conditional_annihilation(dim))
            .tensor(conditional_annihilation(dim))
            .tensor(Operator(np.eye(dim)))
        )
        composite_state.evolve(A_cond)
        AMUT.record_operator_size(BS_0.data, A_cond.data)
        del BS_0, A_cond
    else:
        AMUT.record_operator_size(BS_0.data)
    AMUT.record_state_size(composite_state.data)

    composite_state = composite_state_1.expand(composite_state_2)
    BS_1 = BS.tensor(Operator(np.eye(dim))).tensor(Operator(np.eye(dim)))
    composite_state.evolve(BS_1)

    if lossy:
        A_cond = (
            conditional_annihilation(dim)
            .tensor(conditional_annihilation(dim))
            .tensor(Operator(np.eye(dim)))
            .tensor(Operator(np.eye(dim)))
        )
        composite_state.evolve(A_cond)
        AMUT.record_operator_size(BS_1.data, A_cond.data)
        del BS_1, A_cond
    else:
        AMUT.record_operator_size(BS_1.data)
    AMUT.record_state_size(composite_state.data)

    composite_state = composite_state_1.expand(composite_state_2)
    BS_2 = Operator(np.eye(dim)).tensor(Operator(np.eye(dim))).tensor(BS)
    composite_state.evolve(BS_2)
    if lossy:
        A_cond = (
            Operator(np.eye(dim))
            .tensor(Operator(np.eye(dim)))
            .tensor(conditional_annihilation(dim))
            .tensor(conditional_annihilation(dim))
        )
        composite_state.evolve(A_cond)
        AMUT.record_operator_size(BS_2.data, A_cond.data)
        del BS_2, A_cond
    else:
        AMUT.record_operator_size(BS_2.data)
    AMUT.record_state_size(composite_state.data)

    output = {"state_sizes": AMUT.state_sizes, "operator_sizes": AMUT.operator_sizes}

    print(json.dumps(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "initial_state", type=int, help="Starting individual photon count"
    )
    parser.add_argument(
        "--lossy",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="A boolean argument. Use '--lossy' to set to True, omit for False.",
    )

    args = parser.parse_args()
    lossy_bs_circuit(initial_state=args.initial_state, lossy=args.lossy)
