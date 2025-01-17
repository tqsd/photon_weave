import gc

import psutil
from memory_profiler import profile
from qutip import Qobj, basis, tensor

process = psutil.Process()

STATE_SIZE = 6


# @profile
def run_combine():
    product_state = basis(STATE_SIZE, 0)
    product_state = product_state * product_state.dag()

    for i in range(4):
        c = basis(STATE_SIZE, 0)
        c = c * c.dag()
        product_state = tensor(product_state, c)

    del product_state, c


for i in range(100):
    run_combine()
    gc.collect()
