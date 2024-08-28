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
from typing import Union, List, Tuple, TYPE_CHECKING, Optional

from .expansion_levels import ExpansionLevel
from photon_weave._math.ops import compute_einsum, apply_kraus, kraus_identity_check
from photon_weave.state.exceptions import NotExtractedException
from photon_weave.photon_weave import Config

if TYPE_CHECKING:
    from .envelope import Envelope
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


class Polarization:
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
    __slots__ = (
        "uid", "index", "label", "dimension", "state_vector",
        "density_matrix", "envelope", "expansions", "expansion_level",
        "measured", "__dict__")

    def __init__(
        self,
        polarization: PolarizationLabel = PolarizationLabel.H,
        envelope: Union["Envelope", None] = None,
    ):
        self.uid: uuid.UUID = uuid.uuid4()
        logger.info("Creating polarization with id %s", self.uid)
        self.index: Optional[Union[int, Tuple[int,int]]] = None
        self.label: Optional[PolarizationLabel] = polarization
        self.dimensions: int = 2
        self.state_vector: Optional[jnp.ndarray] = None
        self.density_matrix :Optional[jnp.ndarray] = None
        self.envelope :Optional["Envelope"] = envelope
        self.expansion_level : ExpansionLevel = ExpansionLevel.Label
        self.measured: bool = True

    def __repr__(self) -> str:
        if self.label is not None:
            return f"|{self.label.value}⟩"

        elif self.state_vector is not None:
        # Handle cases where the vector has only one element
            flattened_vector = self.state_vector.flatten()
            formatted_vector: Union[str, List[str]]
            formatted_vector = "\n".join(
                [
                    f"⎢ {''.join([f'{num.real:.2f} {"+" if num.imag >= 0 else "-"} {abs(num.imag):.2f}j' for num in row])} ⎥"
                    for row in self.state_vector
                ]
            )
            formatted_vector = formatted_vector.split("\n")
            formatted_vector[0] = "⎡" + formatted_vector[0][1:-1] + "⎤"
            formatted_vector[-1] = "⎣" + formatted_vector[-1][1:-1] + "⎦"
            formatted_vector = "\n".join(formatted_vector)
            return f"{formatted_vector}"
        elif self.density_matrix is not None:
            formatted_matrix: Union[str,List[str]]
            formatted_matrix = "\n".join(
                [
                    f"⎢ {'   '.join([f'{num.real:.2f} {"+" if num.imag >= 0 else "-"} {abs(num.imag):.2f}j' for num in row])} ⎥"
                    for row in self.density_matrix
                ]
            )

            # Add top and bottom brackets
            formatted_matrix = formatted_matrix.split("\n")
            formatted_matrix[0] = "⎡" + formatted_matrix[0][1:-1] + "⎤"
            formatted_matrix[-1] = "⎣" + formatted_matrix[-1][1:-1] + "⎦"
            formatted_matrix = "\n".join(formatted_matrix)

            return f"{formatted_matrix}"
        elif self.index is not None:
            return "System is part of the Envelope"
        else:
            return "Invalid Fock object" # pragma: no cover

    def expand(self) -> None:
        """
        Expands the representation
        If current representation is label, then it gets
        expanded to state_vector and if it is state_vector
        then it gets expanded to density matrix
        """
        if self.label is not None:
            vector: List[Union[jnp.ndarray, float, complex]]
            match self.label:
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

            self.state_vector = jnp.array(vector)[:, jnp.newaxis]
            self.label = None
            self.expansion_level = ExpansionLevel.Vector
        elif self.state_vector is not None:
            self.density_matrix = jnp.outer(
                self.state_vector.flatten(), jnp.conj(self.state_vector.flatten())
            )
            self.state_vector = None
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
        if self.expansion_level is ExpansionLevel.Matrix and final < ExpansionLevel.Matrix:
            # Check if the state is pure state
            if self.density_matrix is not None:
                state_squared = jnp.matmul(self.density_matrix, self.density_matrix)
            else:
                raise ValueError("Density matrix is None, cannot perform matrix multiplication")
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
            if jnp.allclose(self.state_vector, jnp.array([[1],[0]])):
                self.label = PolarizationLabel.H
                self.state_vector = None
            elif jnp.allclose(self.state_vector, jnp.array([[0],[1]])):
                self.label = PolarizationLabel.V
                self.state_vector = None
            elif jnp.allclose(self.state_vector, jnp.array([[1/jnp.sqrt(2)],[1j/jnp.sqrt(2)]])):
                self.label = PolarizationLabel.R
                self.state_vector = None
            elif jnp.allclose(self.state_vector, jnp.array([[1/jnp.sqrt(2)],[-1j/jnp.sqrt(2)]])):
                self.label = PolarizationLabel.L
                self.state_vector = None

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
        self.label = None
        self.density_matrix = None
        self.state_vector = None

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
        if all(p is None for p in [self.label, self.state_vector, self.density_matrix]):
            if major >= 0:
                self.index = (major, minor)
            else:
                self.index = minor
        else:
            raise NotExtractedException("Polarization state does not seem to be extracted")

    def apply_operation(self, operation: PolarizationOperation, contract:bool = True) -> None:
        """
        Applies a specific operation to the state

        Todo
        ----
        If the state is in the product space the operation should be
        Routed to the correct space

        Parameters
        ----------
        operation: PolarizationOperation
            Operation which should be carried out on this state
        """
        from photon_weave.operation.polarization_operations import (
            PolarizationOperationType,
        )

        match operation.operation:
            case PolarizationOperationType.I:
                return
            case PolarizationOperationType.X:
                if self.label is not None:
                    match self.label:
                        case PolarizationLabel.H:
                            self.label = PolarizationLabel.V
                        case PolarizationLabel.V:
                            self.label = PolarizationLabel.H
                        case PolarizationLabel.R:
                            self.label = PolarizationLabel.L
                        case PolarizationLabel.L:
                            self.label = PolarizationLabel.R
        min_expansion_level = operation.expansion_level_required(self)
        while self.expansion_level < min_expansion_level:
            self.expand()
        operation.compute_operator()
        self._execute_apply(operation)
        if contract:
            self.contract()

    def _execute_apply(self, operation: PolarizationOperation) -> None:
        """
        Internal function, which applies the operator to the state
        Parameters
        ----------
        operation: PolarizationOperation
            The operation which is applied to the state
        """
        if self.expansion_level == 1:
            self.state_vector = compute_einsum("ij,j->i",operation.operator, self.state_vector)
        elif self.expansion_level == 2:
            self.density_matrix = compute_einsum(
                "ik,kj,jl->il",
                operation.operator,
                self.density_matrix,
                operation.operator.T)

    def apply_kraus(self, operators: List[Union[np.ndarray, jnp.ndarray]], identity_check:bool=True) -> None:
        """
        Apply Kraus operators to the state.
        State is automatically expanded to the density matrix representation
        Parameters
        ----------
        operators: List[Union[np.ndarray, jnp.Array]]
            List of the operators
        identity_check: bool
            Signal to check whether or not the operators sum up to identity, True by default
        """

        while self.density_matrix is None:
            self.expand()

        dim = self.density_matrix.shape[0]
        for op in operators:
            if op.shape != (dim, dim):
                raise ValueError(f"Kraus operator has incorrect dimensions: {op.shape}, expected ({dim},{dim})")

        if not kraus_identity_check(operators):
            raise ValueError("Kraus operators do not sum to the identity")
            
        self.density_matrix = apply_kraus(self.density_matrix, operators)
        self.contract()

    def _set_measured(self, **kwargs):
        """
        Internal method, called after measurement,
        it will destroy the state.
        """
        self.measured = True
        self.label = None
        self.expansion_level = None
        self.state_vector = None
        self.density_matrix = None

    def measure(self, separate_measurement:bool=False, **kwargs) -> Union[int,None]:
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
        if self.index is None:
            # Measure in this state
            C = Config()
            if self.expansion_level == ExpansionLevel.Label:
                self.expand()
            if self.expansion_level == ExpansionLevel.Vector:
                assert self.state_vector is not None, "self.state_vector should not be None"
                prob_0 = jnp.abs(self.state_vector[0])**2
                prob_1 = jnp.abs(self.state_vector[1])**2
                assert jnp.isclose(prob_0 + prob_1, 1.0)
                probs = jnp.array([prob_0, prob_1])
                key = jax.random.PRNGKey(C.random_seed)
                result = jax.random.choice(key, a=jnp.array([0,1]), p=probs.ravel())
                self._set_measured()
                return int(result)
            elif self.expansion_level == ExpansionLevel.Matrix:
                # Extract the diagonal elements
                assert self.density_matrix is not None, "self.density_matrix should not be None"
                probabilities = jnp.diag(self.density_matrix).real
                # Normalize
                probabilities = probabilities / jnp.sum(probabilities)
                # Generate a random key
                key = jax.random.PRNGKey(C.random_seed)
                result = jax.random.choice(
                    key,
                    a=jnp.arange(self.density_matrix.shape[0]),
                    p=probabilities
                )
                self._set_measured()
                return int(result)
        # TODO IF MEASURED WHILE IN PRODUCT STATE IS SHOULD ALSO WORK
        return None # pragme: no cover

    def measure_POVM(self, operators:List[Union[np.ndarray, jnp.ndarray]]) -> int:
        """
        Positive Operation-Valued Measurement

        Parameters
        ----------
        *operators: Union[np.ndarray, jnp.Array]
            

        Returns
        -------
        int
            The index of the measurement outcome
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
        #TODO Measuring while in product space should also work
        return -1 # pragma: no cover
        
