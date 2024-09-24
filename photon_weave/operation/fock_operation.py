"""
Operations on fock spaces
"""

from enum import Enum
from typing import Any

import jax.numpy as jnp

from photon_weave._math.ops import (
    annihilation_operator,
    creation_operator,
    displacement_operator,
    phase_operator,
    squeezing_operator,
)
from photon_weave.extra import interpreter
from photon_weave.operation.helpers.fock_dimension_esitmation import FockDimensions
from photon_weave.state.expansion_levels import ExpansionLevel


class FockOperationType(Enum):
    """
    Fock Operation Types
    first value in tuple signals that the state
    should be normalized after operation, second
    element is a list of required elements

    Notes
    -----
    Last element in Tuples is required (to be unique),
    because if two tuples are the same it is assigned
    the same pointer and comparisons don't work then
    """

    Creation = (True, [], ExpansionLevel.Vector, 1)
    Annihilation = (True, [], ExpansionLevel.Vector, 2)
    PhaseShift = (False, ["phi"], ExpansionLevel.Vector, 3)
    Squeeze = (True, ["zeta"], ExpansionLevel.Vector, 4)
    Displace = (False, ["alpha"], ExpansionLevel.Vector, 5)
    Identity = (False, [], ExpansionLevel.Vector, 6)
    Custom = (False, [], ExpansionLevel.Vector, 7)
    Expresion = (False, ["expr"], ExpansionLevel.Vector, 8)

    def __init__(
        self,
        renormalize: bool,
        required_params: list,
        required_expansion_level: ExpansionLevel,
        op_id: int,
    ) -> None:
        self.renormalize = renormalize
        self.required_params = required_params
        self.required_expansion_level = required_expansion_level

    def compute_operator(self, dimensions: int, **kwargs: Any):
        """
        Generates the operator for this opration, given
        the dimensions

        Parameters
        ----------
        dimensions: int
            The dimensions of the state
        **kwargs: Any
            List of key word arguments for specific operator types
        """
        match self:
            case FockOperationType.Creation:
                return creation_operator(dimensions)
            case FockOperationType.Annihilation:
                return annihilation_operator(dimensions)
            case FockOperationType.PhaseShift:
                return phase_operator(dimensions, kwargs["phi"])
            case FockOperationType.Displace:
                return displacement_operator(dimensions, kwargs["alpha"])
            case FockOperationType.Squeeze:
                return squeezing_operator(dimensions, kwargs["zeta"])
            case FockOperationType.Identity:
                return jnp.identity(dimensions)
            case FockOperationType.Expresion:
                context = {
                    "a" : annihilation_operator(dimensions),
                    "a_dag": creation_operator(dimensions)
                }
                context["n"] = jnp.dot(context["a"], context["a_dag"])
                return interpreter(kwargs["expr"], context)

    def compute_dimensions(
        self,
        num_quanta: int,
        state: jnp.ndarray,
        threshold: float = 1 - 1e-6,
        **kwargs: Any,
    ) -> int:
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
        from photon_weave.operation.operation import Operation

        match self:
            case FockOperationType.Creation:
                return int(num_quanta + 2)
            case FockOperationType.Annihilation:
                return num_quanta + 2
            case FockOperationType.PhaseShift:
                return num_quanta + 1
            case FockOperationType.Displace:
                fd = FockDimensions(
                    state,
                    Operation(FockOperationType.Displace, **kwargs),
                    num_quanta,
                    threshold,
                )
                return fd.compute_dimensions()
            case FockOperationType.Squeeze:
                fd = FockDimensions(
                    state,
                    Operation(FockOperationType.Squeeze, **kwargs),
                    num_quanta,
                    threshold,
                )
                return fd.compute_dimensions()
            case FockOperationType.Identity:
                return num_quanta + 1
            case FockOperationType.Expresion:
                fd = FockDimensions(
                    state,
                    Operation(FockOperationType.Expresion, **kwargs),
                    num_quanta,
                    threshold
                )
                return fd.compute_dimensions()
