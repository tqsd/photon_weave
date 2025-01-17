from photon_weave.state.envelope import Envelope
from photon_weave.operation.operation import Operation
from photon_weave.operation.polarization_operation import PolarizationOperationType


class DenseEncoder:
    def encode(self, message: tuple[int, int], env: Envelope) -> Envelope:
        """
        Encode two bits into entangled photon
        """
        op_x = Operation(PolarizationOperationType.X)
        op_z = Operation(PolarizationOperationType.Z)

        ce = env.composite_envelope.states[0]
        match message:
            case (0, 0):
                pass
            case (0, 1):
                env.polarization.apply_operation(op_x)
            case (1, 0):
                env.polarization.apply_operation(op_z)
            case (1, 1):
                env.polarization.apply_operation(op_x)
                env.polarization.apply_operation(op_z)

        return env
