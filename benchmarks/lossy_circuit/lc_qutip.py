import numpy as np
from scipy.linalg import expm
from qutip import basis, destroy, tensor, qeye, Qobj
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

    return Qobj(op_matrix)


def lossy_bs_circuit(initial_state, lossy):
    AMUT = ArrayMemoryUsageTracker()

    # We need enough dimensions to capture all photons in any one state
    dim = 4 * (initial_state + 1)

    env1 = basis(dim, initial_state)
    env2 = basis(dim, initial_state)
    env3 = basis(dim, initial_state)
    env4 = basis(dim, initial_state)

    # Record the initial size
    AMUT.record_state_size(env1.full(), env2.full(), env3.full(), env4.full())
    AMUT.record_operator_size()

    # First Beamsplitter
    a1 = tensor(destroy(dim), qeye(dim))
    a2 = tensor(qeye(dim), destroy(dim))

    H = a1.dag() * a2 + a1 * a2.dag()
    BS = (1j * np.pi / 4 * H).expm()

    # Second Beamsplitter

    composite_state_1 = (BS * tensor(env1, env2)).unit()
    if lossy:
        LOSS = tensor(conditional_annihilation(dim), conditional_annihilation(dim))
        composite_state_1 = (LOSS * composite_state_1).unit()
        AMUT.record_operator_size(BS.full(), LOSS.full())
    else:
        AMUT.record_operator_size(BS.full())

    AMUT.record_state_size(composite_state_1.full(), env3.full(), env4.full())

    composite_state_2 = (BS * tensor(env3, env4)).unit()
    if lossy:
        composite_state_2 = (LOSS * composite_state_2).unit()
        AMUT.record_operator_size(BS.full(), LOSS.full())
    else:
        AMUT.record_operator_size(BS.full())
    AMUT.record_state_size(composite_state_1.full(), composite_state_2.full())

    composite_state_3 = tensor(composite_state_1, composite_state_2)

    # Qutip doesn't have built in way of applying the operator to subspace, thus we need to pad it
    BS_1 = tensor(qeye(dim), BS, qeye(dim))
    composite_state_3 = (BS_1 * composite_state_3).unit()
    if lossy:
        LOSS = tensor(
            qeye(dim),
            conditional_annihilation(dim),
            conditional_annihilation(dim),
            qeye(dim),
        )
        composite_state_3 = (LOSS * composite_state_3).unit()
        AMUT.record_operator_size(BS_1.full(), LOSS.full())
    else:
        AMUT.record_operator_size(BS_1.full())
    AMUT.record_state_size(composite_state_3.full())

    BS_1 = tensor(BS, qeye(dim), qeye(dim))
    composite_state_3 = (BS_1 * composite_state_3).unit()
    if lossy:
        LOSS = tensor(
            conditional_annihilation(dim),
            conditional_annihilation(dim),
            qeye(dim),
            qeye(dim),
        )
        composite_state_3 = (LOSS * composite_state_3).unit()
        AMUT.record_operator_size(BS_1.full(), LOSS.full())
    else:
        AMUT.record_operator_size(BS_1.full())
    AMUT.record_state_size(composite_state_3.full())

    BS_1 = tensor(qeye(dim), qeye(dim), BS)
    composite_state_3 = (BS_1 * composite_state_3).unit()
    if lossy:
        LOSS = tensor(
            qeye(dim),
            qeye(dim),
            conditional_annihilation(dim),
            conditional_annihilation(dim),
        )
        composite_state_3 = (LOSS * composite_state_3).unit()
        AMUT.record_operator_size(BS_1.full(), LOSS.full())
    else:
        AMUT.record_operator_size(BS_1.full())
    AMUT.record_state_size(composite_state_3.full())
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
