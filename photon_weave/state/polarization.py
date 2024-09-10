"""
Polarization State
"""
from __future__ import annotations

from enum import Enum
import logging 
import numpy as np
import jax.numpy as jnp
import jax
import uuid
from typing import Union, List, Tuple, TYPE_CHECKING, Optional, Dict

from .expansion_levels import ExpansionLevel
from .base_state import BaseState
from photon_weave._math.ops import compute_einsum, apply_kraus, kraus_identity_check
from photon_weave.state.exceptions import NotExtractedException
from photon_weave.photon_weave import Config

if TYPE_CHECKING:
    from .envelope import Envelope
    from .composite_envelope import CompositeEnvelope
    from photon_weave.operation.polarization_operations import PolarizationOperation
    
logger = logging.getLogger()


class PolarizationLabel(Enum):
    """
    Labels for the polarization basis states
    """
    H = "H"
    V = "V"
    R = "R"
    L = "L"


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
    def __init__(
        self,
        polarization: PolarizationLabel = PolarizationLabel.H,
        envelope: Union["Envelope", None] = None,
    ):
        self.uid: uuid.UUID = uuid.uuid4()
        logger.info("Creating polarization with uid %s", self.uid)
        self.index: Optional[Union[int, Tuple[int,int]]] = None
        self.state: Optional[Union[jnp.ndarray, PolarizationLabel]] = polarization
        self._dimensions: int = 2
        self.envelope :Optional["Envelope"] = envelope
        self.expansion_level : ExpansionLevel = ExpansionLevel.Label
        self.measured: bool = False
        self.composite_envelope = None
        

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions:int) -> None:
        raise ValueError("Dimensions can not be set for Polarization type, 2 by default")

    def expand(self) -> None:
        """
        Expands the representation
        If current representation is label, then it gets
        expanded to state_vector and if it is state_vector
        then it gets expanded to density matrix
        """
        # If the state is in composite envelope expand the product space there
        if isinstance(self.index, tuple) or isinstance(self.index, list):
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            self.composite_envelope.contract(self)
            return
        # If the state is in envelope expand the product space there
        elif isinstance(self.index, int):
            assert isinstance(self.envelope, Envelope)
            self.envelope.contract()
            return

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
            self.state = jnp.array(vector)[:, jnp.newaxis]
            self.expansion_level = ExpansionLevel.Vector
        elif self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            self.state = jnp.dot(self.state, self.state.T)
            self.expansion_level = ExpansionLevel.Matrix

    def contract(self, final:ExpansionLevel = ExpansionLevel.Label, tol:float=1e-6) -> None:
        """
        Attempts to contract the representation to the level defined in `final`argument.

        Parameters
        ----------
        final: ExpansionLevel
            Expected expansion level after contraction
        tol: float
            Tolerance when comparing matrices
        """
        # If state is in composite envelope conract product state there
        if isinstance(self.index, tuple) or isinstance(self.index, list):
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            self.composite_envelope.contract(self)
            return
        # If state is in envelope conract product state there
        elif isinstance(self.index, int):
            assert isinstance(self.envelope, Envelope)
            self.envelope.contract()
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
                assert isinstance(self.state, jnp.ndarray)
                phase = jnp.exp(-1j * jnp.angle(self.state[0]))
                self.state = self.state*phase
                self.expansion_level = ExpansionLevel.Vector
        if self.expansion_level is ExpansionLevel.Vector and final < ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            if jnp.allclose(self.state, jnp.array([[1],[0]])):
                self.state = PolarizationLabel.H
            elif jnp.allclose(self.state, jnp.array([[0],[1]])):
                self.state = PolarizationLabel.V
            elif jnp.allclose(self.state, jnp.array([[1/jnp.sqrt(2)],[1j/jnp.sqrt(2)]])):
                self.state = PolarizationLabel.R
            elif jnp.allclose(self.state, jnp.array([[1/jnp.sqrt(2)],[-1j/jnp.sqrt(2)]])):
                self.state = PolarizationLabel.L
            self.expansion_level = ExpansionLevel.Label

    def extract(self, index: Union[int, Tuple[int, int]]) -> None:
        """
        This method is called, when the state is
        joined into a product space. Then the
        index is set and the label, density_matrix and
        state_vector is set to None
        Parameters:
        index: Union[int, Tuple[int, int]
            Index of the state in the product state
        """
        self.index = index
        self.state = None

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
        if self.state is None:
            if major >= 0:
                self.index = (major, minor)
            else:
                self.index = minor
        else:
            raise NotExtractedException("Polarization state does not seem to be extracted")

    def _set_measured(self) -> None:
        """
        Internal method, called after measurement,
        it will destroy the state.
        """
        self.measured = True
        self.state = None
        self.index = None

    def measure(self, destructive:bool = True, separate_measurement:bool=False) -> Dict[BaseState, int]:
        """
        Measures this state. If the state is not in a product state it will
        produce a measurement, otherwise it will return None.
        
        Parameters
        ----------
        separate_measurement: bool

        Returns
        ----
        Union[int,None]
            Measurement Outcome
        """
        # If the state is in the envelope, measure there
        if isinstance(self.index, int):
            assert isinstance(self.envelope, Envelope)
            return self.envelope.measure(self, separate_measurement=separate_measurement, destructive=destructive)

        # If the state is in the composite envelope, measure there
        if isinstance(self.index, tuple) or isinstance(self.index, list):
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            return self.composite_envelope.measure(self)

        results: Dict[BaseState, int] = {}
        C = Config()
        if self.expansion_level == ExpansionLevel.Label:
            self.expand()
        if self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            prob_0 = jnp.abs(self.state[0])**2
            prob_1 = jnp.abs(self.state[1])**2
            assert jnp.isclose(prob_0 + prob_1, 1.0)
            probs = jnp.array([prob_0[0], prob_1[0]])
            key = C.random_key
            outcome = jax.random.choice(key, a=jnp.array([0,1]), p=probs.ravel())
            results[self] = int(outcome)
        elif self.expansion_level == ExpansionLevel.Matrix:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, self.dimensions)
            probabilities = jnp.diag(self.state).real
            probabilities = probabilities / jnp.sum(probabilities)
            # Generate a random key
            key = C.random_key
            outcome = jax.random.choice(
                key,
                a=jnp.arange(self.state.shape[0]),
                p=probabilities
            )
            results[self] = int(outcome)
        if results[self] == 0:
            self.state = PolarizationLabel.H
        elif results[self] == 1:
            self.state = PolarizationLabel.V
        self.expansion_level = ExpansionLevel.Label
        if destructive:
            self._set_measured()
        return results

    def measure_POVM(self, operators:List[Union[np.ndarray, jnp.ndarray]], destructive:bool=True) -> Tuple[int, Dict[BaseState, int]]:
        """
        Positive Operation-Valued Measurement

        Parameters
        ----------
        operators: Union[np.ndarray, jnp.Array]

        Returns
        -------
        int
            The index of the measurement outcome
        """
        if isinstance(self.index, int):
            assert isinstance(self.envelope, Envelope)
            return self.envelope.measure_POVM(operators, self)
        elif isinstance(self.index, tuple) or isinstance(self.index, list):
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            return self.composite_envelope.measure_POVM(operators, self, destructive=destructive)

        while self.expansion_level < ExpansionLevel.Matrix:
            self.expand()

        assert isinstance(self.state, jnp.ndarray)
        assert self.state.shape == (self.dimensions, self.dimensions)

        # Compute probabilities p(i) = Tr(E_i * rho) for each POVM operator E_i
        probabilities = jnp.array([
            jnp.trace(jnp.matmul(op, self.state)).real for op in operators
        ])

        # Normalize probabilities (handle numerical issues)
        probabilities = probabilities / jnp.sum(probabilities)

        # Generate a random key
        C = Config()
        key = C.random_key

        # Sample the measurement outcome
        outcome = int(jax.random.choice(
            key,
            a=jnp.arange(len(operators)),
            p=probabilities
        ))

        self.state = jnp.matmul(
            operators[outcome], jnp.matmul(
                self.state,jnp.conj(operators[outcome].T)))
        self.state = self.state / jnp.trace(self.state)
        
        if destructive:
            self._set_measured()

        return (outcome, {})

