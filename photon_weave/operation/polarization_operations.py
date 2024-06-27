from enum import Enum, auto

import numpy as np
import scipy.linalg

from .generic_operation import GenericOperation


class PolarizationOperationType(Enum):
    I = auto()
    X = auto()
    Y = auto()
    Z = auto()
    H = auto()
    S = auto()
    T = auto()
    PS = auto()
    RX = auto()
    RY = auto()
    RZ = auto()
    U3 = auto()
    Custom = auto()


class PolarizationOperation(GenericOperation):
    def __init__(
        self, operation: PolarizationOperationType, apply_count: int = 1, **kwargs
    ):
        self.kwargs = kwargs
        self.operation = operation
        match operation:
            case PolarizationOperationType.RX:
                if "theta" not in self.kwargs:
                    raise KeyError("The argument 'theta' is required for RX gate")
            case PolarizationOperationType.RY:
                if "theta" not in self.kwargs:
                    raise KeyError("The argument 'theta' is required for RY gate")
            case PolarizationOperationType.RZ:
                if "theta" not in self.kwargs:
                    raise KeyError("The argument 'theta' is required for RZ gate")
            case PolarizationOperationType.U3:
                if (
                    "theta" not in self.kwargs
                    or "lambda" not in self.kwargs
                    or "phi" not in self.kwargs
                ):
                    raise KeyError(
                        "The arguments 'theta', 'lambda' and 'phi' are required for RZ gate"
                    )
            case PolarizationOperationType.PS:
                if "sigma" not in self.kwargs:
                    raise KeyError(
                        "The argument 'phi' is required for PS gate"
                    )
            case PolarizationOperationType.Custom:
                if "operator" not in self.kwargs:
                    raise KeyError(
                        "The argument 'operator' is required for PS gate"
                    )
                if self.kwargs.operator.shape() != (2,2):
                    raise KeyError(
                        "Custom operator must be 2by2 matrix"
                    )

    def compute_operator(self):
        match self.operation:
            case PolarizationOperationType.I:
                self.operator = np.eye(2, dtype=np.complex128)
            case PolarizationOperationType.X:
                self.operator = np.zeros((2, 2), dtype=np.complex128)
                self.operator[0][1] = 1
                self.operator[1][0] = 1
            case PolarizationOperationType.Y:
                self.operator = np.zeros((2, 2), dtype=np.complex128)
                self.operator[0][1] = -1j
                self.operator[1][0] = 1j
            case PolarizationOperationType.Z:
                self.operator = np.zeros((2, 2), dtype=np.complex128)
                self.operator[0][0] = 1
                self.operator[1][1] = -1
            case PolarizationOperationType.H:
                self.operator = 1/np.sqrt(2)*np.array([[1, 1], [1, -1]], dtype=np.complex_)
            case PolarizationOperationType.S:
                self.operator = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
            case PolarizationOperationType.T:
                self.operator = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex_)
            case PolarizationOperationType.PS:
                self.operator = np.array(
                    [[1, 0], [0, np.exp(1j*self.kwargs['phi'])]],
                    dtype=np.complex_)
            case PolarizationOperationType.RX:
                sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
                self.operator = scipy.linalg.expm(-1j*theta*sigma_x/2)
            case PolarizationOperationType.RY:
                sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
                self.operator = scipy.linalg.expm(-1j*theta*sigma_y/2)
            case PolarizationOperationType.RZ:
                sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
                self.operator = scipy.linalg.expm(-1j*theta*sigma_z/2)
            case PolarizationOperationType.U3:
                theta = self.kwargs["theta"]
                lambda_ = self.kwargs["lambda"]
                phi = self.kwargs["phi"]
                self.operator = np.array(
                    [
                        [np.cos(theta / 2), -np.exp(1j * lambda_) * np.sin(theta / 2)],
                        [
                            np.exp(1j * phi) * np.sin(theta / 2),
                            np.exp(1j * (phi + lambda_)) * np.cos(theta / 2),
                        ],
                    ],
                    dtype=complex,
                )
            case PolarizationOperationType.Custom:
                self.operator = self.kwargs["operator"]

    def expansion_level_required(self, state) -> int:
        from photon_weave.state.polarization import PolarizationLabel

        match self.operation:
            case PolarizationOperationType.I:
                return 0
            case PolarizationOperationType.X:
                if state.label is not None:
                    return 0
                return 1
            case PolarizationOperationType.Y:
                return 1
            case _:
                return 1
