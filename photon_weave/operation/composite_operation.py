
from enum import Enum
from typing import Any, List
import jax.numpy as jnp
from jax.scipy.linalg import expm


from photon_weave._math.ops import (
    creation_operator,
    annihilation_operator
)
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.base_state import BaseState
from photon_weave.state.fock import Fock

class CompositeOperationType(Enum):

    NonPolarizingBeamSplitter = (True, ["eta"], [Fock, Fock], ExpansionLevel.Vector, 1)

    def __init__(self, renormalize: bool, required_params: list,
                 expected_base_state_types: List[BaseState],
                 required_expansion_level: ExpansionLevel,
                 op_id) -> None:
        
        self.renormalize = renormalize
        self.required_params = required_params
        self.expected_base_state_types = expected_base_state_types
        self.required_expansion_level = required_expansion_level

    def compute_operator(self, dimensions:List[int], **kwargs: Any):
        """
        Generates the operator for this operation, given
        the dimensions

        Parameters
        ----------
        dimensions: List[int]
            The dimensions of the spaces
        **kwargs: Any
            List of key word arguments for specific operator types
        """
        match self:
            case CompositeOperationType.NonPolarizingBeamSplitter:
                a = creation_operator(dimensions[0])
                a_dagger = annihilation_operator(dimensions[0])
                b = creation_operator(dimensions[1])
                b_dagger = annihilation_operator(dimensions[1])

                operator = (
                    jnp.kron(a_dagger,b) + jnp.kron(a, b_dagger)
                    )
                return expm(1j*kwargs["eta"]*operator)

    def compute_dimensions(
            self,
            num_quanta: List[int],
            states: jnp.ndarray,
            threshold: float = 1 - 1e6,
            **kwargs: Any
    ) -> List[int]:
        """
        Compute the dimensions for the operator. Application of the
        operator could change the dimensionality of the space. For
        example creation operator, would increase the dimensionality
        of the space, if the prior dimensionality doesn't account
        for the new particle.

        Parameters
        ----------
        num_quanta: int
            Number of particles in the space currently. In other words
            highest basis with non_zero probability in the space
        state: jnp.ndarray
            Traced out state, usede for final dimension estimation
        threshold: float
            Minimal amount of state that has to be included in the
            post operation state
        **kwargs: Any
            Additional parameters, used to define the operator

        Returns
        -------
        int
           New number of dimensions

        Notes
        -----
        This functionality is called before the operator is computed, so that
        the dimensionality of the space can be changed before the application
        and the dimensionality of the operator and space match
        """
        match self:
            case CompositeOperationType.NonPolarizingBeamSplitter:
                dim = int(jnp.sum(jnp.array(num_quanta))) + 1
                return [dim,dim]
