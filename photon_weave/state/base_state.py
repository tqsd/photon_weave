from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union, Tuple, TYPE_CHECKING
import numpy as np
import jax.numpy as jnp
from uuid import UUID

from photon_weave._math.ops import kraus_identity_check, apply_kraus
from photon_weave.state.expansion_levels import ExpansionLevel

if TYPE_CHECKING:
    from photon_weave.state.polarization import Polarization, PolarizationLabel

class BaseState(ABC):

    __slots__ = (
        "label", "_uid", "_state_vector", "density_matrix",
        "__dict__", "_expansion_level", "_index", "_dimensions",
        "_measured"
    )

    @property
    def uid(self) -> Union[str, UUID]:
        return self._uid

    @uid.setter
    def uid(self, state_vector: Union[UUID, str]) -> None:
        self._uid= uid

    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass

    @property
    def expansion_level(self) -> int:
        return self._expansion_level

    @expansion_level.setter
    def expansion_level(self, expansion_level: 'ExpansionLevel') -> None:
        self._expansion_level = expansion_level

    @property
    def index(self) -> Union[None, int, Tuple[int, int]]:
        return self._index

    @index.setter
    def index(self, index : Union[None, int, Tuple[int, int]]) -> None:
        self._index = index

    @property
    def dimensions(self) -> Union[int, 'PolarizationLabel']:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: int) -> None:
        self._dimensions = dimensions 

    # Dunder methods
    def __hash__(self):
        return hash(self.uid)

    def __repr__(self) -> str:
        if self.label is not None:
            if isinstance(self.label, Enum):
                return f"|{self.label.value}⟩"
            elif isinstance(self.label, int):
                return f"|{self.label}⟩"

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
        else:
            return str(self.uid)

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

    @abstractmethod
    def expand(self) -> None:
        pass


    @abstractmethod
    def _set_measured(self):
        pass

    @abstractmethod
    def extract(self, index:Union[int, Tuple[int, int]]) -> None:
        pass

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
