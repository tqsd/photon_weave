"""
Operations on fock spaces
"""

from numba import uintc
from enum import Enum, auto
import jax.numpy as jnp
from typing import Optional, Any
from scipy.stats import norm

from photon_weave._math.ops import (
    annihilation_operator,
    creation_operator,
    displacement_operator,
    squeezing_operator,
    phase_operator
)
from photon_weave.extra import interpreter
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
    # TESTED
    Creation = (True, [], ExpansionLevel.Vector, 1)
    # TESTED
    Annihilation = (True, [], ExpansionLevel.Vector, 2)
    PhaseShift = (False, ['phi'], ExpansionLevel.Vector, 3)
    Squeeze = (True, ['alpha'], ExpansionLevel.Vector, 4)
    Displace = (False, ['zeta'], ExpansionLevel.Vector, 5)
    Identity = (False, [], ExpansionLevel.Vector, 6)
    Custom = (False, [], ExpansionLevel.Vector, 7)
    Expresion = (False, ["expr"], ExpansionLevel.Vector, 8)

    def __init__(self, renormalize:bool, required_params: list,
                 required_expansion_level: ExpansionLevel, op_id:int) -> None:
        self.renormalize = renormalize
        self.required_params = required_params
        self.required_expansion_level=required_expansion_level

    def compute_operator(self, dimensions:int, **kwargs:Any):
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
                return creation_operator(uintc(dimensions))
            case FockOperationType.Annihilation:
                return annihilation_operator(uintc(dimensions))
            case FockOperationType.PhaseShift:
                return phase_operator(uintc(dimensions), kwargs["phi"])
            case FockOperationType.Displace:
                return displacement_operator(uintc(dimensions), kwargs["alpha"])
            case FockOperationType.Squeeze:
                return squeezing_operator(uintc(dimensions), kwargs["zeta"])
            case FockOperationType.Identity:
                return jnp.identity(dimensions)
            case FockOperationType.Expresion:
                return interpreter(dimensions, kwargs["expr"])

        
    def compute_dimensions(self, num_quanta:int) -> int:
        match self:
            case FockOperationType.Creation:
                return int(num_quanta + 2)
            case FockOperationType.Annihilation:
                return num_quanta + 2
            case FockOperationType.PhaseShift:
                return num_quanta + 1
            case FockOperationType.Displace:
                a_squared = kwargs["alpha"]**2
                return jnp.ceil(a_squared + num_quanta + 3 * jnp.sqrt(a_squared + num_quanta))
            case FockOperationType.Squeeze:
                mean_increase = jnp.sinh(kwargs["zeta"])
                total_mean_photon_number = num_quanta +  mean_increase
                std_dev_photon_number = jnp.sqrt(2*mean_increase*(mean_increase+1))
                n_max = int(jnp.ceil(total_mean_photon_number+3*std_dev_photon_number))
                cumulative_prob = 0.0
                threshold = 0.99
                while cumulative_prob < threshold:
                    cumulative_prob = norm.cdf(
                        n_max,
                        loc=total_mean_photon_number,
                        scale=std_dev_photon_number
                    )
                    n_max += 1
                return n_max
            case FockOperationType.Identity:
                return num_quanta + 1
            case FockOperationType.Expresion:
                return num_quanta + 1


