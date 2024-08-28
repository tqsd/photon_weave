"""
Polarization State
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import jax.numpy as jnp

from .expansion_levels import ExpansionLevel
from photon_weave._math.ops import compute_einsum
from typing import Union, List


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
        "index", "label", "dimension", "state_vector",
        "density_matrix", "envelope", "expansions", "expansion_level",
        "measured", "__dict__")
    )
    def __init__(
        self,
        polarization: PolarizationLabel = PolarizationLabel.H,
        envelope: "Envelope" = None,
    ):
        self.index = None
        self.label = polarization
        self.dimensions = 2
        self.state_vector = None
        self.density_matrix = None
        self.envelope = envelope
        self.expansion_level = ExpansionLevel.Label
        self.measured = True

    def __repr__(self) -> str:
        if self.label is not None:
            return f"|{self.label.value}⟩"
        elif self.state_vector is not None:
            formatted_vector = "\n".join(
                [
                    f"{complex_num.real:.2f} {'+' if complex_num.imag >= 0 else '-'} {abs(complex_num.imag):.2f}j"
                    for complex_num in self.state_vector.flatten()
                ]
            )

            return f"{formatted_vector}"
        elif self.density_matrix is not None:
            formatted_matrix = "\n".join(
                [
                    "\t".join(
                        [
                            f"({num.real:.2f} {'+' if num.imag >= 0 else '-'} {abs(num.imag):.2f}j)"
                            for num in row
                        ]
                    )
                    for row in self.density_matrix
                ]
            )
            return f"{formatted_matrix}"
        elif self.index is not None:
            return "System is part of the Envelope"
        else:
            return "Invalid Fock object"

    def _get_state(self, label):

    def expand(self) -> None:
        """
        Expands the representation
        If current representation is label, then it gets
        expanded to state_vector and if it is state_vector
        then it gets expanded to density matrix
        """
        if self.label is not None:
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


    def contract(self, tol:float=1e-7) -> None:
        """
        Tests if the state can be contracted and if
        so it contracts its representation.
        If expansion_level is Density Matrix and the state is
        pure state it contracts it to the vector representation.
        If the state is in one of the basis states, than it
        contracts it to the 
        """
        if self.expansion_level is ExpansionLevel.Matrix:
            # Check if the state is pure state
            state_squared = jnp.matmul(self.state, self.state)
            state_trace = jnp.trace(state_squared)
            if jnp.abs(state_trace-1) < tol:
                # The state is pure
                eigenvalue, eigenvectors = jnp.linalg.eigh(self.state)
                pure_state_index = jnp.argmax(jnp.abs(eigenvalues -1.0) < tol)
                self.state_vector = [:, pure_state_index]
                self.density_matrix = None
                self.expansion_level = ExpansionLevel.Vector
        if self.expansion_level is ExpansionLevel.Vector:
            match self.state_vector:
                case jnp.allclose(self.state_vector, jnp.array([[1],[0]])):
                    self.label = Polarization.H
                    self.state_vector = None
                case jnp.allclose(self.state_vector, jnp.array([[0],[1]])):
                    self.label = Polarization.V
                    self.state_vector = None
                case jnp.allclose(self.state_vector, jnp.array([[1/jnp.sqrt(2)],[1j/jnp.sqrt(2)]])):
                    self.label = Polarization.R
                    self.state_vector = None
                case jnp.allclose(self.state_vector, jnp.array([[1/jnp.sqrt(2)],[-1j/jnp.sqrt(2)]])):
                    self.label = Polarization.L
                    self.state_vector = None

    def extract(self, index: int) -> None:
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

    def apply_operation(self, operation: PolarizationOperation) -> None:
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
            if self.expansion_level == ExpansionLevel.Label:
                self.expand()
            if self.expansion_level == ExpansionLevel.Vector:
                prob_0 = jnp.abs(self.state[0])**2
                prob_1 = jnp.abs(self.state[1])**2
                assert jnp.isclose(prob_0 + prob_1, 1.0)
                probs = jnp.array([prob_0, prob_1])
                key = jnp.random.PRNGKey(jax.random.default_prng_seed())
                result = jax.random.choice(key, a=jnp.array([0,1]), p=probs)
                self._set_measured()
                return result
            elif self.expansion_level == ExpansionLevel.Matrix:
                # Extract the diagonal elements
                probabilities = jnp.diag(self.state).real
                # Normalize
                probabilities = probabilities / jnp.sum(probabilities)
                # Generate a random key
                key = jnp.random.PRNGKey(jax.random.default_prng_seed())
                measurement_result = jax.random.choice(
                    key,
                    a=jnp.arrange(self.state.shape[0]),
                    probabilities = probabilities
                )
                self._set_measured()
                return result

        return None

    def POVM_measurement(self, *operators:Union[np.ndarray, jnp.Array]) -> int:
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

        if self.expansion_level == ExpansionLevel.Label:
            self.expand()

        if self.expansion_level == ExpansionLevel.Vector:
            self.expand()

        # Compute probabilities p(i) = Tr(E_i * rho) for each POVM operator E_i
        probabilities = jnp.array([
            jnp.trace(jnp.matmul(op, self.state)).real for op in operators
        ])

        # Normalize probabilities (handle numerical issues)
        probabilities = probabilities / jnp.sum(probabilities)

        # Generate a random key
        key = jax.random.PRNGKey(jax.random.default_prng_seed())

        # Sample the measurement outcome
        measurement_result = jax.random.choice(
            key,
            a=jnp.arrange(len(operators)),
            p=probabilities
        )
        self._set_measured()
        return measurement_result
        
