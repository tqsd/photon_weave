from photon_weave.state.envelope import Envelope
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.operation.polarization_operation import PolarizationOperationType
from photon_weave.operation.operation import Operation
from photon_weave.operation.composite_operation import CompositeOperationType


class EntangledPhotonSource:
    def emit(self) -> tuple[Envelope, Envelope]:
        env1 = Envelope()
        env2 = Envelope()
        env1.polarization.expand()
        env2.polarization.expand()

        env1.fock.state = 1
        env2.fock.state = 1

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)

        # Entangle the polarizations
        op_h = Operation(PolarizationOperationType.H)
        op_cnot = Operation(CompositeOperationType.CXPolarization)

        ce.apply_operation(op_h, env1.polarization)
        ce.apply_operation(op_cnot, env1.polarization, env2.polarization)

        return env1, env2
