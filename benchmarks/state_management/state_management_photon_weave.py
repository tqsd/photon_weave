import gc

import jax
import psutil
from jax import config

from photon_weave.operation import Operation, PolarizationOperationType
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.custom_state import CustomState
from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock

# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
process = psutil.Process()
STATE_SIZE = 6


# config.update("jax_log_compiles", True)
# @profile
def run_combine():
    C = CustomState(STATE_SIZE)
    C.expand()
    C.expand()
    ce = CompositeEnvelope(C)
    for i in range(4):
        c = CustomState(STATE_SIZE)
        ce = CompositeEnvelope(ce, c)
        ce.combine(C, c)

    jax.block_until_ready(ce.product_states[0].state)
    CompositeEnvelope._containers = {}
    CompositeEnvelope._instances = {}


for i in range(100):
    run_combine()
    gc.collect()
