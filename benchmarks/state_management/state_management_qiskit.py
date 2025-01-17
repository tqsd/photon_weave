import gc

import numpy as np
import psutil
from qiskit.quantum_info import DensityMatrix

process = psutil.Process()

STATE_SIZE = 6


def run_combine():
    # Create a 6-dimensional "basis vector" for |0>
    basis_vec = np.zeros(STATE_SIZE, dtype=complex)
    basis_vec[0] = 1.0

    # Make an outer product to form the density matrix: |0><0|
    product_state = DensityMatrix(np.outer(basis_vec, basis_vec.conjugate()))

    # Repeat four times
    for i in range(4):
        # Another basis vector for c
        c_vec = np.zeros(STATE_SIZE, dtype=complex)
        c_vec[0] = 1.0
        c_dm = DensityMatrix(np.outer(c_vec, c_vec.conjugate()))

        # Tensor (Kronecker) product
        product_state = product_state.tensor(c_dm)

    # Clean up references
    del product_state, c_dm, basis_vec, c_vec


# Example single run
for i in range(100):
    run_combine()
    gc.collect()
