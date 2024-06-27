"""
Operations on fock spaces
"""
from enum import Enum, auto

import numpy as np
from scipy.linalg import expm

from photon_weave.extra import interpreter

from .generic_operation import GenericOperation


class FockOperationType(Enum):
    # implemented
    Creation = auto()
    # implemented
    Annihilation = auto()
    # implemented
    PhaseShift = auto()
    # implemented
    Squeeze = auto()
    # implemented
    Displace = auto()
    Identity = auto()
    Custom = auto()


class FockOperation(GenericOperation):
    def __init__(self, operation: FockOperationType, apply_count: int = 1, **kwargs):
        self.kwargs = kwargs
        self.operation = operation
        self.operator = None
        self.apply_count = apply_count
        # For the case of applying the ladder operators to the state directly
        self.renormalize = None
        # Ladder operators are not unitary, in our case we normalize after applying
        match self.operation:
            case FockOperationType.Creation:
                self.renormalize = True
            case FockOperationType.Annihilation:
                self.renormalize = True
            case FockOperationType.PhaseShift:
                if "phi" not in kwargs:
                    raise KeyError(
                        "The 'phi' argument is required for Phase Shift operator"
                    )
            case FockOperationType.Displace:
                self.renormalize = True
                if "alpha" not in kwargs:
                    raise KeyError(
                        "The 'alpha' argument is required for Displace operator"
                    )
            case FockOperationType.Squeeze:
                if "zeta" not in kwargs:
                    raise KeyError(
                        "The compley 'zeta' argument is required for Squeeze operator"
                    )

    def compute_operator(self, dimensions):
        """
        Computes the correct operator with the appropariate
        dimensions.
        """
        match self.operation:
            case FockOperationType.Creation:
                self.operator = self._create(dimensions)
            case FockOperationType.Annihilation:
                self.operator = self._destroy(dimensions)
            case FockOperationType.PhaseShift:
                n = np.arange(dimensions)
                phases = np.exp(1j * n * self.kwargs["phi"])
                self.operator = np.diag(phases)
            case FockOperationType.Displace:
                alpha = self.kwargs["alpha"]
                self.operator = expm(
                    alpha * self._create(dimensions) - self._destroy(dimensions) * alpha
                )
            case FockOperationType.Squeeze:
                zeta = self.kwargs["zeta"]
                a = self._destroy(dimensions)
                a_dagger = self._create(dimensions)
                self.operator = expm(
                    0.5
                    * (np.conj(zeta) * np.dot(a, a) - zeta * np.dot(a_dagger, a_dagger))
                )
            case FockOperationType.Identity:
                self.operator = np.eye(dimensions)
            case FockOperationType.Custom:
                if "expression" in self.kwargs:
                    self._evaluate_custom_operator(
                        self.kwargs["expression"], dimensions
                    )
        if self.apply_count > 1:
            self.operator = np.linalg.matrix_power(self.operator, self.apply_count)

    def _create(self, cutoff):
        a_dagger = np.zeros((cutoff, cutoff), dtype=np.complex_)
        for i in range(1, cutoff):
            a_dagger[i, i - 1] = np.sqrt(i)
        return a_dagger

    def _destroy(self, cutoff):
        a = np.zeros((cutoff, cutoff), dtype=np.complex_)
        for i in range(1, cutoff):
            a[i - 1, i] = np.sqrt(i)
        return a

    def expansion_level_required(self) -> int:
        """
        Returns the expansion level required
        """
        match self.operation:
            case FockOperationType.Creation:
                return 0
            case FockOperationType.Annihilation:
                return 0
            case FockOperationType.PhaseShift:
                return 1
            case FockOperationType.Displace:
                return 1
            case FockOperationType.Squeeze:
                return 1
            case _:
                return 1

    def cutoff_required(self, num_quanta=0) -> int:
        """
        Returns the expansion level required
        """
        match self.operation:
            case FockOperationType.Displace:
                return int(np.ceil(4 * np.abs(self.kwargs["alpha"]) ** 2))
            case FockOperationType.Squeeze:
                r = np.abs(self.kwargs["zeta"])
                return int(2 + 4 * r + 2 * r**4)
            case _:
                return 0

    def assign_operator(self, expression):
        if self.operation is FockOperationType.Custom:
            self.expression = expression

    def _evaluate_custom_operator(self, expression, dimensions):
        context = {
            "a": self._destroy(dimensions),
            "a_dag": self._create(dimensions),
        }
        context["n"] = np.dot(context["a_dag"], context["a"])
        self.operator = interpreter(expression, context)
