from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union, Tuple
import numpy as np
import jax.numpy as jnp
from uuid import UUID

from photon_weave._math.ops import kraus_identity_check, apply_kraus

class BaseState(ABC):

    __slots__ = ("label", "_uid", "_state_vector", "density_matrix", "__dict__", "_expansion_level", "_index")

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
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass

    @property
    @abstractmethod
    def expansion_level(self) -> int:
        pass

    @expansion_level.setter
    def expansion_level(self, expansion_level: 'ExpansionLevel') -> None:
        self._expansion_level = expansion_level

    @property
    @abstractmethod
    def index(self) -> Union[None, int, Tuple[int, int]]:
        pass

    @index.setter
    def index(self, index : Union[None, int, Tuple[int, int]]) -> None:
        self._index = expansion_level
    @property

    @abstractmethod
    def state_vector(self) -> jnp.ndarray:
        pass

    @state_vector.setter
    def state_vector(self, state_vector: jnp.ndarray) -> None:
        self._state_vector = state_vector

    
    @abstractmethod
    def uid(self) -> jnp.ndarray:
        pass

    @state_vector.setter
    def uid(self, state_vector: Union[UUID, str]) -> None:
        self._uid= uid


    @abstractmethod
    def _set_measured(self):
        pass

    @abstractmethod
    def extract(self, index:Union[int, Tuple[int, int]]) -> None:
        pass
