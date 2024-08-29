"""
Fock state 
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax
import uuid
from typing import TYPE_CHECKING, Optional, Tuple, List, Union, Any

from photon_weave.photon_weave import Config
from photon_weave.operation.fock_operation import FockOperation, FockOperationType
from photon_weave._math.ops import (
    normalize_matrix, normalize_vector, num_quanta_matrix, num_quanta_vector, apply_kraus, kraus_identity_check
)

from .envelope import EnvelopeAssignedException
from .expansion_levels import ExpansionLevel
from .base_state import BaseState

if TYPE_CHECKING:
    from .envelope import Envelope



class Fock(BaseState):
    """
    Fock class

    This class handles the Fock state or points to the
    Envelope or Composite envelope, which holds the state

    Attributes
    ----------
    index: Union[int, Tuple[int]]
        If Fock space is part of a product space index
        holds information about the space and subspace index
        of this state
    dimension: int
        The dimensions of the Hilbert space, can be set or is
        computed on the fly when expanding the state
    label: int
        If expansion level is Label then label holds the state
        (number basis state)
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
    """
    __slots__ = (
        "uid", "index", "label", "dimension", "state_vector",
        "density_matrix", "envelope", "expansions", "expansion_level",
        "measured")

    def __init__(self, envelope: Optional[Envelope] = None):
        self.uid: uuid.UUID = uuid.uuid4()
        self.index: Optional[Union[int, Tuple[int, int]]] = None
        self.dimensions: int = -1
        self.label: Optional[int] = 0
        self.state_vector: Optional[jnp.ndarray] = None
        self.density_matrix: Optional[jnp.ndarray] = None
        self.envelope: Optional["Envelope"] = envelope
        self.expansion_level : ExpansionLevel = ExpansionLevel.Label
        self.measured = False

    def __eq__(self, other: Any):
        """
        Comparison operator for the states, returns True if
        states are expanded to the same level and are not part
        of the product space
        Todo
        ----
        Method should work for a spaces if they do not have equal
        expansion level
        """
        if not isinstance(other, Fock):
            return False
        if self.label is not None and other.label is not None:
            if self.label == other.label:
                return True
            return False
        if self.state_vector is not None and other.state_vector is not None:
            if np.array_equal(self.state_vector, other.state_vector):
                return True
            return False
        if self.density_matrix is not None and other.density_matrix is not None:
            if np.array_equal(self.density_matrix, other.density_matrix):
                return True
            return False
        return False

    def expand(self):
        """
        Expands the representation. If the state is stored in
        label then it is expanded to state_vector and if the
        state is in state_vector, then the state is expanded
        to the state_matrix
        """
        if self.dimensions < 0:
            self.dimensions = self.label + 3
        if self.expansion_level is ExpansionLevel.Label:
            state_vector = np.zeros(int(self.dimensions))
            state_vector[self.label] = 1
            self.state_vector = state_vector[:, np.newaxis]
            self.label = None
            self.expansion_level = ExpansionLevel.Vector
        elif self.expansion_level is ExpansionLevel.Vector:
            self.density_matrix = np.outer(
                self.state_vector.flatten(), np.conj(self.state_vector.flatten())
            )
            self.state_vector = None
            self.expansion_level = ExpansionLevel.Matrix

    def contract(self, final: ExpansionLevel = ExpansionLevel.Label, tol:float=1e-6) -> None:
        """
        Attempts to contract the representation to the level defined in `final`argument.

        Parameters
        ----------
        final: ExpansionLevel
            Expected expansion level after contraction
        tol: float
            Tolerance when comparing matrices
        """
        if self.expansion_level is ExpansionLevel.Matrix and final < ExpansionLevel.Matrix:
            # Check if the state is pure state
            assert self.density_matrix is not None, "Density matrix should not be None"
            state_squared = jnp.matmul(self.density_matrix, self.density_matrix)
            state_trace = jnp.trace(state_squared)
            if jnp.abs(state_trace-1) < tol:
                # The state is pure
                eigenvalues, eigenvectors = jnp.linalg.eigh(self.density_matrix)
                pure_state_index = jnp.argmax(jnp.abs(eigenvalues -1.0) < tol)
                assert pure_state_index is not None, "pure_state_index should not be None"
                self.state_vector = eigenvectors[:, pure_state_index].reshape(-1,1)
                # Normalizing the phase
                assert self.state_vector is not None, "self.state_vector should not be None"
                phase = jnp.exp(-1j * jnp.angle(self.state_vector[0]))
                self.state_vector = self.state_vector*phase
                self.density_matrix = None
                self.expansion_level = ExpansionLevel.Vector
        if self.expansion_level is ExpansionLevel.Vector and final < ExpansionLevel.Vector:
            assert self.state_vector is not None, "self.state_vector should not be None"
            ones = jnp.where(self.state_vector == 1)[0]
            if ones.size == 1:
                self.label = int(ones[0])
                self.state_vector = None
                self.expansion_level = ExpansionLevel.Label

    def extract(self, index: Union[int, Tuple[int, int]]) -> None:
        """
        This method is called, when the state is
        joined into a product space. Then the
        index is set and the label, density_matrix and
        state_vector is set to None
        """
        self.index = index
        self.label = None
        self.density_matrix = None
        self.state_vector = None

    def apply_operation(self, operation: FockOperation) -> None:
        """
        Applies a specific operation to the state

        Todo
        ----
        If the state is in the product space the operation should be
        Routed to the correct space

        Parameters
        ----------
        operation: FockOperation
            Operation which should be carried out on this state
        """
        match operation.operation:
            case FockOperationType.Creation:
                if self.label is not None:
                    self.label += operation.apply_count
                    return
            case FockOperationType.Annihilation:
                if self.label is not None:
                    self.label -= operation.apply_count
                    if self.label < 0:
                        self.label = 0
                    return
        min_expansion_level = operation.expansion_level_required()
        while self.expansion_level < min_expansion_level:
            self.expand()

        cutoff_required = operation.cutoff_required(self._num_quanta)
        if cutoff_required > self.dimensions:
            self.resize(cutoff_required)

        match operation.operation:
            case FockOperationType.Creation:
                if self._num_quanta + operation.apply_count + 1 > self.dimensions:
                    self.resize(self._num_quanta + operation.apply_count + 1)
        operation.compute_operator(self.dimensions)

        self._execute_apply(operation)

    @property
    def _num_quanta(self) -> int:
        """
        The highest possible measurement outcome.
        returns highest basis with non_zero probability

        Returns:
        --------
        int
            Highest possible measurement outcome, -1 if failed
        """
        if self.label is not None:
            return self.label
        if self.state_vector is not None:
            return num_quanta_vector(self.state_vector)
        if self.density_matrix is not None:
            return num_quanta_matrix(self.density_matrix)
        return -1

    def _execute_apply(self, operation: FockOperation):
        """
        Actually executes the operation

        Todo
        ----
        Consider using gpu for this operation
        """
        assert operation.operator is not None, "operation.operators must not be None"
        if self.state_vector is not None:
            self.state_vector = operation.operator @ self.state_vector
        if self.density_matrix is not None:
            self.density_matrix = operation.operator @ self.density_matrix
            self.density_matrix @= operation.operator.conj().T

        if operation.renormalize:
            self.normalize()
    def normalize(self) -> None:
        """
        Normalizes the state.
        """
        if self.density_matrix is not None:
            self.density_matrix = normalize_matrix(self.density_matrix)
        elif self.state_vector is not None:
            self.state_vector = normalize_vector(self.state_vector)

    def resize(self, new_dimensions:int) -> bool:
        """
        Resizes the state to the new_dimensions

        Parameters
        ----------
        new_dimensions: int
            New size to change to

        Returns
        -------
        bool
            True if resizes was successful
        """
        success = False
        if self.label is not None:
            self.dimensions = new_dimensions
            success = True
        elif self.dimensions < new_dimensions:
            pad_size = new_dimensions - self.dimensions
            if self.state_vector is not None:
                self.state_vector = jnp.pad(
                    self.state_vector,
                    ((0, pad_size), (0, 0)),
                    "constant",
                    constant_values=(0,),
                )
                success = True
            if self.density_matrix is not None:
                self.density_matrix = jnp.pad(
                    self.density_matrix,
                    ((0, pad_size), (0, pad_size)),
                    "constant",
                    constant_values=0,
                )
                success = True
        # Truncate
        elif self.dimensions > new_dimensions:
            pad_size = new_dimensions - self.dimensions
            if self.state_vector is not None:
                if jnp.all(self.state_vector[new_dimensions:] == 0):
                    self.state_vector = self.state_vector[:new_dimensions]
                    success = True
            if self.density_matrix is not None:
                bottom_rows_zero = jnp.all(self.density_matrix[new_dimensions:, :] == 0)
                right_columns_zero = np.all(
                    self.density_matrix[:, new_dimensions:] == 0
                )
                if bottom_rows_zero and right_columns_zero:
                    self.density_matrix = self.density_matrix[
                        :new_dimensions, :new_dimensions
                    ]
                    success = True
        self.dimensions = new_dimensions
        return success

    def set_index(self, minor:int, major:int=-1):
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
        if major >= 0:
            self.index = (major, minor)
        else:
            self.index = minor

    def measure(self, non_destructive=False, remove_composite=True, partial=False) -> int:
        """
        Measures the state in the number basis. This Method can be used if the
        state resides in the Envelope or Composite Envelope

        Parameters
        ----------
        non_destructive: bool
            If True the state won't be destroyed post measurement
        remove_composite: bool
            If True and the state is part of the composite envelope
            the state won't be removed from the composite 
        partial: bool
            If true then accompanying Polarization space in the envelop
            won't be measured

        Returns
        -------
        outcome: int
            Outcome of the measurement, -1 if failed
        """
        if self.measured:
            raise FockAlreadyMeasuredException()
        result:int = -1
        if isinstance(self.index, int):
            assert self.envelope is not None, "Envelope should not be None"
            return self.envelope.measure(remove_composite=remove_composite)
        else:
            C = Config()
            match self.expansion_level:
                case ExpansionLevel.Label:
                    assert self.label is not None, "self.label should not be None"
                    result = self.label
                case ExpansionLevel.Vector:
                    assert self.state_vector is not None, "self.state_vector should not be None"
                    probs = jnp.abs(self.state_vector.flatten()) ** 2
                    probs = probs.ravel()
                    assert jnp.isclose(sum(probs), 1)
                    key = jax.random.PRNGKey(C.random_seed)
                    result = int(jax.random.choice(
                        key,
                        a=jnp.array(
                            list(range(len(probs)))),
                        p=probs))
                case ExpansionLevel.Matrix:
                    assert self.density_matrix is not None, "self.density_matrix should not be None"
                    probs = jnp.diag(self.density_matrix).real
                    probs = probs / jnp.sum(probs)
                    key = jax.random.PRNGKey(C.random_seed)
                    result = int(jax.random.choice(
                        key,
                        a=jnp.arange(self.density_matrix.shape[0]),
                        p=probs
                    ))
        if not partial and self.envelope:
            self.envelope._set_measured(remove_composite=remove_composite)
        self._set_measured()
        return int(result)

    def measure_POVM(self, operators:List[Union[np.ndarray, jnp.ndarray]]) -> int:
        """
        Positive Operation-Valued Measurement

        Parameters
        ----------
        *operators: Union[np.ndarray, jnp.Array]
            

        Returns
        -------
        int
            The index of the measurement outcome, -1 if measurement failed
        """
        if self.index is None:
            if self.expansion_level == ExpansionLevel.Label:
                self.expand()

            if self.expansion_level == ExpansionLevel.Vector:
                self.expand()

            # Compute probabilities p(i) = Tr(E_i * rho) for each POVM operator E_i
            assert self.density_matrix is not None, "self.density_matrix should not be None"
            probabilities = jnp.array([
                jnp.trace(jnp.matmul(op, self.density_matrix)).real for op in operators
            ])

            # Normalize probabilities (handle numerical issues)
            probabilities = probabilities / jnp.sum(probabilities)

            # Generate a random key
            C = Config()
            key = jax.random.PRNGKey(C.random_seed)

            # Sample the measurement outcome
            measurement_result = jax.random.choice(
                key,
                a=jnp.arange(len(operators)),
                p=probabilities
            )
            self._set_measured()
            return int(measurement_result)
        # TODO IF MEASURED WHILE IN PRODUCT STATE IT SHOULD ALSO WORK
        return -1

    def _set_measured(self, **kwargs):
        """
        Destroys the state
        """
        self.measured = True
        self.label = None
        self.expansion_level = None
        self.state_vector = None
        self.density_matrix = None
        self.index = None

    def get_subspace(self) -> Union[int,jnp.ndarray]:
        """
        Returns the space subspace. If the state is in label representation
        then it is expanded once. If the state is in product space,
        then the space will be traced from the product space

        Returns
        -------
        state: Union[np.array]
            The state in the numpy array, or -1 if failed
        """
        if self.index is None:
            if not self.label is None:
                return self.label
            if not self.state_vector is None:
                return self.state_vector
            elif not self.density_matrix is None:
                return self.density_matrix
        elif isinstance(self.index, int):
            # State is in the Envelope
            pass

        elif isinstance(self.index, tuple):
            assert self.envelope is not None, "self.envelope should not be None"
            assert self.envelope.composite_envelope is not None, "self.envelope.composite_envelope should not be None"
            state = self.envelope.composite_envelope._trace_out(self, destructive=False)
            return state
        return -1



class FockAlreadyMeasuredException(Exception):
    pass
