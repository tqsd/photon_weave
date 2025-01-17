"""
Polarization State
"""

from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp

from photon_weave.operation import PolarizationOperationType
from photon_weave.photon_weave import Config

from .base_state import BaseState
from .expansion_levels import ExpansionLevel
from .utils.measurements import measure_matrix, measure_vector
from .utils.operations import apply_operation_matrix, apply_operation_vector
from .utils.routing import route_operation
from .utils.state_transform import state_contract, state_expand

if TYPE_CHECKING:
    from photon_weave.operation import Operation

    from .composite_envelope import CompositeEnvelope
    from .envelope import Envelope

logger = logging.getLogger()


class PolarizationLabel(Enum):
    """
    Labels for the polarization basis states
    """

    H = "H"
    V = "V"
    R = "R"
    L = "L"
    A = "A"
    D = "D"


class Polarization(BaseState):
    """
    Polarization class

    This class handles the polarization state or points to the
    Envelope or Composite envelope, which holds the state

    Attributes
    ----------
    index: Union[int, Tuple[int]]
        If polarization is part of a product space index
        holds information about the space and subspace index
        of this state
    dimension: int
        The dimensions of the Hilbert space (2)
    label: PolarizationLabel
        If expansion level is Label then label holds the state
    state_vector: np.array
        If expansion level is Vector then state_vector holds
        the state
    density_matrix: np.array
        If expansion level is Matrix then density_matrix holds
        the state
    envelope: Envelope
        If the state is part of a envelope, the envelope attribute
        holds a reference to the Envelope instance
    expansion_level: ExpansionLevel
        Holds information about the expansion level of this system
    measured: bool
        If the state was measured than measured is True
    """

    __slots__ = BaseState.__slots__

    def __init__(
        self,
        polarization: PolarizationLabel = PolarizationLabel.H,
        envelope: Union["Envelope", None] = None,
    ):
        self.uid: uuid.UUID = uuid.uuid4()
        logger.info("Creating polarization with uid %s", self.uid)
        self.index: Optional[Union[int, Tuple[int, int]]] = None
        self.state: Optional[Union[jnp.ndarray, PolarizationLabel]] = polarization
        self._dimensions: int = 2
        self.envelope: Optional["Envelope"] = envelope
        self._expansion_level: Optional[ExpansionLevel] = ExpansionLevel.Label
        self._measured: bool = False
        self._composite_envelope: Optional[CompositeEnvelope] = None

    @property
    def measured(self) -> bool:
        return self._measured

    @measured.setter
    def measured(self, measured: bool) -> None:
        self._measured = measured

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: int) -> None:
        raise ValueError(
            "Dimensions can not be set for Polarization type, 2 by default"
        )

    @route_operation()
    def expand(self) -> None:
        """
        Expands the representation
        If current representation is label, then it gets
        expanded to state_vector and if it is state_vector
        then it gets expanded to density matrix

        Notes
        -----
        Method is decorated with route_operation. If the state is
        contained in the product state, the corresponding operation
        will be executed in the state container, which contains this
        state.
        """
        if self.expansion_level == ExpansionLevel.Label:
            assert isinstance(self.state, PolarizationLabel)
            vector: List[Union[jnp.ndarray, float, complex]]
            match self.state:
                case PolarizationLabel.H:
                    vector = [1, 0]
                case PolarizationLabel.V:
                    vector = [0, 1]
                case PolarizationLabel.R:
                    # Right circular polarization = (1/sqrt(2)) * (|H⟩ + i|V⟩)
                    vector = [1 / jnp.sqrt(2), 1j / jnp.sqrt(2)]
                case PolarizationLabel.L:
                    # Left circular polarization = (1/sqrt(2)) * (|H⟩ - i|V⟩)
                    vector = [1 / jnp.sqrt(2), -1j / jnp.sqrt(2)]
                case PolarizationLabel.A:
                    vector = [1 / jnp.sqrt(2), -1 / jnp.sqrt(2)]
                case PolarizationLabel.D:
                    vector = [1 / jnp.sqrt(2), 1 / jnp.sqrt(2)]
            self.state = jnp.array(vector, dtype=jnp.complex128)[:, jnp.newaxis]
            self.expansion_level = ExpansionLevel.Vector
        else:
            assert isinstance(self.state, jnp.ndarray)
            assert isinstance(self.expansion_level, ExpansionLevel)
            self.state, self.expansion_level = state_expand(
                self.state, self.expansion_level, self.dimensions
            )

    @route_operation()
    def contract(
        self, final: ExpansionLevel = ExpansionLevel.Label, tol: float = 1e-6
    ) -> None:
        """
        Attempts to contract the representation to the level defined in final argument.

        Parameters
        ----------
        final: ExpansionLevel
            Expected expansion level after contraction
        tol: float
            Tolerance when comparing matrices

        Notes
        -----
        Method is decorated with route_operation. If the state is
        contained in the product state, the corresponding operation
        will be executed in the state container, which contains this
        state.
        """
        # If state was measured, then do nothing
        if self.measured:
            return

        if (
            self.expansion_level is ExpansionLevel.Matrix
            and final < ExpansionLevel.Matrix
        ):
            assert isinstance(self.state, (jnp.ndarray, int))
            self.state, self.expansion_level, success = state_contract(  # type: ignore
                self.state, self.expansion_level
            )
        if (
            self.expansion_level is ExpansionLevel.Vector
            and final < ExpansionLevel.Vector
        ):
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            if jnp.allclose(self.state, jnp.array([[1], [0]])):
                self.state = PolarizationLabel.H
            elif jnp.allclose(self.state, jnp.array([[0], [1]])):
                self.state = PolarizationLabel.V
            elif jnp.allclose(
                self.state, jnp.array([[1 / jnp.sqrt(2)], [1j / jnp.sqrt(2)]])
            ):
                self.state = PolarizationLabel.R
            elif jnp.allclose(
                self.state, jnp.array([[1 / jnp.sqrt(2)], [-1j / jnp.sqrt(2)]])
            ):
                self.state = PolarizationLabel.L
            elif jnp.allclose(
                self.state, jnp.array([[1 / jnp.sqrt(2)], [-1 / jnp.sqrt(2)]])
            ):
                self.state = PolarizationLabel.A
            elif jnp.allclose(
                self.state, jnp.array([[1 / jnp.sqrt(2)], [1 / jnp.sqrt(2)]])
            ):
                self.state = PolarizationLabel.D
            if isinstance(self.state, PolarizationLabel):
                self.expansion_level = ExpansionLevel.Label

    def extract(self, index: Union[int, Tuple[int, int]]) -> None:
        """
        This method is called, when the state is joined into a product space. Then the
        index is set and the label, density_matrix and state_vector is set to None.

        Parameters:

        index: Union[int, Tuple[int, int]
            Index of the state in the product state
        """
        self.index = index
        self.state = None

    def set_index(self, minor: int, major: int = -1) -> None:
        """
        Sets the index, when product space is created, or
        manipulated

        Parameters
        ----------
        minor: int
            Minor index show the order of tensoring in the space
        major: int
            Major index points to the product space when it is in
            CompositeEnvelope
        """
        if self.state is None:
            if major >= 0:
                self.index = (major, minor)
            else:
                self.index = minor
        else:
            raise ValueError("Polarization state does not seem to be extracted")

    def _set_measured(self) -> None:
        """
        Internal method, called after measurement,
        it will destroy the state.
        """
        self.measured = True
        self.state = None
        self.index = None
        self.expansion_level = None

    @route_operation()
    def measure(
        self, separate_measurement: bool = False, destructive: bool = True
    ) -> Dict[BaseState, int]:
        """
        Measures this state. If the state is not in a product state it will
        produce a measurement, otherwise it will return None.

        Parameters
        ----------
        separate_measurement: bool

        Returns
        -------
        Union[int,None]
            Measurement Outcome

        Notes
        -----
        Method is decorated with route_operation. If the state is
        contained in the product state, the corresponding operation
        will be executed in the state container, which contains this
        state.
        """
        if self.expansion_level == ExpansionLevel.Label:
            self.expand()

        match self.expansion_level:
            case ExpansionLevel.Vector:
                assert isinstance(self.state, jnp.ndarray)
                outcomes, post_measurement_state = measure_vector(
                    [self], [self], self.state
                )
            case ExpansionLevel.Matrix:
                assert isinstance(self.state, jnp.ndarray)
                outcomes, post_measurement_state = measure_matrix(
                    [self], [self], self.state
                )
        # Reconstruct the state post measurement
        if outcomes[self] == 0:
            self.state = PolarizationLabel.H
        elif outcomes[self] == 1:
            self.state = PolarizationLabel.V
        self.expansion_level = ExpansionLevel.Label

        if destructive:
            self._set_measured()

        return outcomes

    @route_operation()
    def apply_operation(self, operation: Operation) -> None:
        """
        Applies an operation to the state. If state is in some product
        state, the operator is correctly routed to the specific
        state

        Parameters
        ----------
        operation: Operation
            Operation with operation type: PolarizationOperationType

        Notes
        -----
        Method is decorated with route_operation. If the state is
        contained in the product state, the corresponding operation
        will be executed in the state container, which contains this
        state.
        """
        assert isinstance(operation._operation_type, PolarizationOperationType)
        assert isinstance(self.expansion_level, ExpansionLevel)

        while self.expansion_level < operation.required_expansion_level:
            self.expand()

        operation.compute_dimensions(0, jnp.array([0]))

        match self.expansion_level:
            case ExpansionLevel.Vector:
                assert isinstance(self.state, jnp.ndarray)
                self.state = apply_operation_vector(
                    [self], [self], self.state, operation.operator
                )
            case ExpansionLevel.Matrix:
                assert isinstance(self.state, jnp.ndarray)
                self.state = apply_operation_matrix(
                    [self], [self], self.state, operation.operator
                )

        assert isinstance(self.state, jnp.ndarray)
        if not jnp.any(jnp.abs(self.state) > 0):
            raise ValueError(
                "The state is entirely composed of zeros, is |0⟩ "
                "attempted to be anniilated?"
            )
        if operation.renormalize:
            self.state = self.state / jnp.linalg.norm(self.state)

        C = Config()
        if C.contractions:
            self.contract()
