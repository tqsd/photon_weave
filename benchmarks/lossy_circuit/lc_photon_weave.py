import jax.numpy as jnp
import jax
import numpy as np
from photon_weave.state.envelope import Envelope
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.operation import Operation, FockOperationType, CompositeOperationType
import argparse
import json
from array_memory_usage_tracker import ArrayMemoryUsageTracker

INITIAL_STATE = 8
LOSS = 1


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

    return jnp.array(op_matrix)


def lossy_bs_circuit(initial_state, lossy):
    AMUT = ArrayMemoryUsageTracker()

    env1 = Envelope()
    env1.fock.state = initial_state
    env2 = Envelope()
    env2.fock.state = initial_state
    env3 = Envelope()
    env3.fock.state = initial_state
    env4 = Envelope()
    env4.fock.state = initial_state
    for env in [env1, env2, env3, env4]:
        env.fock.expand()

    # Initial Size
    AMUT.record_state_size(env1.fock, env2.fock, env3.fock, env4.fock)
    AMUT.record_operator_size()

    ce = CompositeEnvelope(env1, env2, env3, env4)

    bs = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)

    context = {"a": lambda dims: conditional_annihilation(dims[0])}
    loss = Operation(
        FockOperationType.Expresion, expr=("s_mult", "a", 1), context=context
    )

    ce.apply_operation(bs, env1.fock, env2.fock)
    if lossy:
        env1.fock.apply_operation(loss)
        env2.fock.apply_operation(loss)
        AMUT.record_operator_size(bs, loss)
    else:
        AMUT.record_operator_size(bs)
    AMUT.record_state_size(ce.product_states[0], env3.fock, env4.fock)

    ce.apply_operation(bs, env3.fock, env4.fock)
    if lossy:
        env3.fock.apply_operation(loss)
        env4.fock.apply_operation(loss)
        AMUT.record_operator_size(bs, loss)
    else:
        AMUT.record_operator_size(bs)
    AMUT.record_state_size(ce.product_states[0], ce.product_states[1])

    ce.apply_operation(bs, env2.fock, env3.fock)
    if lossy:
        env2.fock.apply_operation(loss)
        env3.fock.apply_operation(loss)
        AMUT.record_operator_size(bs, loss)
    else:
        AMUT.record_operator_size(bs)
    AMUT.record_state_size(ce.product_states[0])

    ce.apply_operation(bs, env1.fock, env2.fock)
    if lossy:
        env1.fock.apply_operation(loss)
        env2.fock.apply_operation(loss)
        AMUT.record_operator_size(bs, loss)
    else:
        AMUT.record_operator_size(bs)
    AMUT.record_state_size(ce.product_states[0])

    ce.apply_operation(bs, env3.fock, env4.fock)
    if lossy:
        env3.fock.apply_operation(loss)
        env4.fock.apply_operation(loss)
        AMUT.record_operator_size(bs, loss)
    else:
        AMUT.record_operator_size(bs)
    AMUT.record_state_size(ce.product_states[0])

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
