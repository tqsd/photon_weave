"""
Fock state 
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax
import uuid
from typing import TYPE_CHECKING, Optional, Tuple, List, Union, Any, Dict

from photon_weave.photon_weave import Config
from photon_weave.operation.fock_operation import FockOperation, FockOperationType
from photon_weave._math.ops import (
    normalize_matrix, normalize_vector, num_quanta_matrix, num_quanta_vector, apply_kraus, kraus_identity_check
)

from .envelope import EnvelopeAssignedException
from .expansion_levels import ExpansionLevel
from .base_state import BaseState
from photon_weave.state.composite_envelope import CompositeEnvelope

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
        "uid", "index", "label", "dimensions", "state_vector",
        "density_matrix", "envelope", "expansions", "expansion_level",
        "measured")

    def __init__(self, envelope: Optional[Envelope] = None):
        self.uid: uuid.UUID = uuid.uuid4()
        self.index: Optional[Union[int, Tuple[int, int]]] = None
        self.dimensions: int = -1
        self.state = 0
        self.envelope: Optional["Envelope"] = envelope
        self._composite_envelope = None
        self.expansion_level : ExpansionLevel = ExpansionLevel.Label
        self.measured = False

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other: Any) -> bool:
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
        if isinstance(self.state, int) and isinstance(other.state, int):
            if self.state == other.state:
                return True
        elif isinstance(self.state, jnp.ndarray) and isinstance(other.state, jnp.ndarray):
            if self.state.shape == other.state.shape:
                if jnp.allclose(self.state, other.state):
                    return True
        return False

    def expand(self) -> None:
        """
        Expands the representation. If the state is stored in
        label then it is expanded to state_vector and if the
        state is in state_vector, then the state is expanded
        to the state_matrix
        """
        if isinstance(self.index, int):
            assert isinstance(self.envelope, Envelope)
            self.envelope.expand()
        if isinstance(self.index, tuple) or isinstance(self.index, list):
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            self.composite_envelope.expand(self)

        if self.dimensions < 0:
            assert self.dimensions is not None, "self.dimensions shoul not be None"
            if isinstance(self.state, int):
                self.dimensions = self.state + 3

        if self.expansion_level is ExpansionLevel.Label:
            assert isinstance(self.state, int)
            state_vector = jnp.zeros(int(self.dimensions))
            state_vector = state_vector.at[self.state].set(1)
            new_state_vector = state_vector[:, jnp.newaxis]
            self.state = new_state_vector
            self.expansion_level = ExpansionLevel.Vector
        elif self.expansion_level is ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            new_density_matrix = jnp.outer(
                self.state.flatten(), jnp.conj(self.state.flatten())
            )
            self.state = new_density_matrix
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
        # If state was measured, then do nothing
        if self.measured:
            return
        if self.expansion_level is ExpansionLevel.Matrix and final < ExpansionLevel.Matrix:
            # Check if the state is pure state
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, self.dimensions)
            state_squared = jnp.matmul(self.state, self.state)
            state_trace = jnp.trace(state_squared)
            if jnp.abs(state_trace-1) < tol:
                # The state is pure
                eigenvalues, eigenvectors = jnp.linalg.eigh(self.state)
                pure_state_index = jnp.argmax(jnp.abs(eigenvalues -1.0) < tol)
                assert pure_state_index is not None, "pure_state_index should not be None"
                self.state = eigenvectors[:, pure_state_index].reshape(-1,1)
                # Normalizing the phase
                assert self.state is not None, "self.state should not be None"
                phase = jnp.exp(-1j * jnp.angle(self.state[0]))
                self.state= self.state*phase
                self.expansion_level = ExpansionLevel.Vector
        if self.expansion_level is ExpansionLevel.Vector and final < ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            ones = jnp.where(self.state == 1)[0]
            if ones.size == 1:
                self.state = int(ones[0])
                self.expansion_level = ExpansionLevel.Label

    def extract(self, index: Union[int, Tuple[int, int]]) -> None:
        """
        This method is called, when the state is
        joined into a product space. Then the
        index is set and the label, density_matrix and
        state_vector is set to None
        """
        self.index = index
        self.state = None

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
        if isinstance(self.state, int):
            return self.state
        elif isinstance(self.state, jnp.ndarray):
            if self.state.shape == (self.dimensions, 1):
                return num_quanta_vector(self.state)
            elif self.state.shape == (self.dimensions, self.dimensions):
                return num_quanta_matrix(self.state)
        return -1

    def set_index(self, minor:int, major:int=-1) -> None:
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

    def measure(self, destructive:bool=True, separate_measurement:bool=False) -> Dict[BaseState, int]:
        """
        Measures the state in the number basis. This Method can be used if the
        state resides in the Envelope or Composite Envelope

        Parameters
        ----------
        destructive: bool
            If False the state won't be destroyed post measurement (removed)
            The state will still be modified due to measurement
        separate_measurement: bool
            If True and the state is part of the composite envelope
            the state won't be removed from the composite 

        Returns
        -------
        Dict[BaseState, int]
            Dictionary of outcomes
        """
        if isinstance(self.index, int):
            assert isinstance(self.envelope, Envelope)
            return self.envelope.measure(
                self,
                separate_measurement=separate_measurement,
                destructive=destructive
            )
        if isinstance(self.index, tuple) or isinstance(self.index, list):
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            return self.composite_envelope.measure(self)

        if self.index is not None:
            assert self.envelope is not None, "Envelope should not be None"
            return self.envelope.measure()
        C = Config()
        match self.expansion_level:
            case ExpansionLevel.Label:
                assert isinstance(self.state, int)
                result = self.state
            case ExpansionLevel.Vector:
                assert isinstance(self.state, jnp.ndarray)
                assert self.state.shape == (self.dimensions, 1)
                probs = jnp.abs(self.state.flatten()) ** 2
                probs = probs.ravel()
                assert jnp.isclose(sum(probs), 1)
                key = C.random_key
                result = int(jax.random.choice(
                    key,
                    a=jnp.arange(len(probs)),
                    p=probs
                ))
            case ExpansionLevel.Matrix:
                assert isinstance(self.state, jnp.ndarray)
                assert self.state.shape == (self.dimensions, self.dimensions)
                probs = jnp.diag(self.state).real
                probs = probs / jnp.sum(probs)
                key = C.random_key
                result = int(jax.random.choice(
                    key,
                    a=jnp.arange(len(probs)),
                    p=probs
                ))

        self.state = result
        self.expansion_level = ExpansionLevel.Label
        outcomes: Dict[BaseState, int] = {}
        outcomes[self] = int(result)
        if destructive:
            self._set_measured()

        if self.envelope is not None:
            if not self.envelope.polarization.measured:
                out = self.envelope.polarization.measure()
                assert isinstance(out, dict)
                for key, value in out.items():
                    assert isinstance(key, BaseState)
                    assert isinstance(value, int)
                    outcomes[key] = value
        return outcomes


        

    def _set_measured(self, **kwargs: Dict[str, Any]) -> None:
        """
        Destroys the state
        """
        self.measured = True
        self.state = None
        self.index = None
        self.expansion_level = None
