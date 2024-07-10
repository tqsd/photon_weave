"""
Composite Operation:
operates on multiple spaces
"""

from enum import Enum, auto
from typing import Union

import numpy as np

from photon_weave._math.ops import _expm
from photon_weave.operation.fock_operation import FockOperation, FockOperationType
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope


class WrongNumberOfStatesException(Exception):
    pass


class CompositeOperationType(Enum):
    NonPolarizingBeamSplit = auto()
    CNOT = auto()


def ensure_equal_expansion(func):
    """
    Ensures that all given states, have the same expansion level
    """

    def wrapper(self, *states, **kwargs):
        expansion_levels = [s.expansion_level for s in states]
        if max(expansion_levels) == 0:
            desired_expansion = 1
        else:
            desired_expansion = max(expansion_levels)
        for s in states:
            while s.expansion_level < desired_expansion:
                s.expand()
        return func(self, *states, **kwargs)

    return wrapper


def ensure_composite(func):
    def wrapper(self, *states, **kwargs):
        composite_envelopes = [
            s.composite_envelope for s in states if s.composite_envelope is not None
        ]
        composite_envelopes = list(set(composite_envelopes))
        if len(composite_envelopes) == 0:
            new_envelope = CompositeEnvelope(*states)
        elif len(composite_envelopes) == 1:
            existing_envelope = composite_envelopes[0]
            for state in states:
                existing_envelope.add_envelope(state)
        else:
            new_envelope = CompositeEnvelope(*states)
        return func(self, *states, **kwargs)

    return wrapper


class CompositeOperation:
    def __init__(self, operation: CompositeOperationType, apply_num: int = 1, **kwargs):
        self.kwargs = kwargs
        self.operation = operation
        self.operator = None
        self.apply_num = apply_num
        match self.operation:
            case CompositeOperationType.NonPolarizingBeamSplit:
                if "theta" not in kwargs:
                    self.kwargs["theta"] = np.pi / 4

    def operate(self, *args) -> Union[CompositeEnvelope, Envelope]:
        match self.operation:
            case CompositeOperationType.NonPolarizingBeamSplit:
                self._operate_non_polarizing_beam_split(*args)
            case CompositeOperationType.CNOT:
                self._operate_cnot(*args)

    @ensure_composite
    def _operate_cnot(self, *args, **kwargs):
        ce = args[0].composite_envelope
        ce.combine(args[0].polarization, args[1].polarization)
        ce.rearange(args[0].polarization, args[1].polarization)
        self.compute_operator()
        ce._apply_operator(self, args[0].polarization, args[1].polarization)

    @ensure_composite
    def _operate_non_polarizing_beam_split(self, *args, **kwargs):
        if len(args) != 2:
            raise WrongNumberOfStatesException(
                "Beam split can operate only on two states"
            )
        ce = args[0].composite_envelope
        ce.combine(args[0].fock, args[1].fock)
        ce.rearange(args[0].fock, args[1].fock)
        dim1 = args[0].fock.dimensions
        dim2 = args[1].fock.dimensions
        self.compute_operator(dimensions=(dim1, dim2))
        ce._apply_operator(self, args[0].fock, args[1].fock)

    def compute_operator(self, *args, **kwargs):
        match self.operation:
            case CompositeOperationType.NonPolarizingBeamSplit:
                eta = self.kwargs.get(
                    "eta", 1
                )  # Default to complete overlap if not provided

                fo = FockOperation(operation=FockOperationType.Identity)
                dim1, dim2 = kwargs["dimensions"]
                a = fo._create(dim1)
                a_dagger = fo._destroy(dim1)
                b = fo._create(dim2)
                b_dagger = fo._destroy(dim2)

                # Adjusting the operator with the eta parameter
                self.operator = np.sqrt(eta) * (
                    np.kron(a_dagger, b) + np.kron(a, b_dagger)
                )
                # If necessary, include (1 - eta) terms to model distinguishable paths
                # This part is highly dependent on the physical interpretation of eta
                # and the specifics of the HOM effect you wish to model

                theta = self.kwargs.get("theta", 0)  # Default to 0 if not provided
                self.operator = theta * self.operator
                self.operator = _expm(1j * self.operator)
            case CompositeOperationType.CNOT:
                self.operator = np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
                )
