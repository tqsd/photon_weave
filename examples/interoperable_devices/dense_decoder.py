"""
Dense Decoder
"""

from photon_weave.state.envelope import Envelope
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.operation import Operation
from photon_weave.operation.composite_operation import CompositeOperationType
from photon_weave.operation.polarization_operation import PolarizationOperationType


class DenseDecoder:
    def decode(self, env1: Envelope, env2: Envelope) -> tuple[int, int]:
        op_h = Operation(PolarizationOperationType.H)
        op_cnot = Operation(CompositeOperationType.CXPolarization)

        ce = CompositeEnvelope(env1, env2)

        ce.apply_operation(op_cnot, env1.polarization, env2.polarization)

        env1.polarization.apply_operation(op_h)

        m1 = env1.measure()
        m2 = env2.measure()

        # Get the outcomes of the polarization measurements
        p1 = m1[env1.polarization]
        p2 = m2[env2.polarization]

        return p1, p2
