from enum import Enum, auto
from typing import List, Any
import jax.numpy as jnp

from photon_weave._math.ops import (
    identity_operator,
    x_operator,
    y_operator,
    z_operator,
    s_operator,
    t_operator,
    sx_operator,
    rx_operator,
    ry_operator,
    rz_operator,
    u3_operator
)
from photon_weave.state.expansion_levels import ExpansionLevel


class PolarizationOperationType(Enum):
    """
    PolarizationOperationType

    Constructs an operator, which acts on a single Polarization Space
    """
    I  = (True, [], ExpansionLevel.Vector, 1) 
    X  = (True, [], ExpansionLevel.Vector, 2) 
    Y  = (True, [], ExpansionLevel.Vector, 3) 
    Z  = (True, [], ExpansionLevel.Vector, 4) 
    H  = (True, [], ExpansionLevel.Vector, 5) 
    S  = (True, [], ExpansionLevel.Vector, 6) 
    T  = (True, [], ExpansionLevel.Vector, 7) 
    SX = (True, [], ExpansionLevel.Vector, 8) 
    RX = (True, ["theta"], ExpansionLevel.Vector, 9) 
    RY = (True, ["theta"], ExpansionLevel.Vector, 10) 
    RZ = (True, ["theta"], ExpansionLevel.Vector, 11) 
    U3 = (True, ["phi", "theta", "omega"], ExpansionLevel.Vector, 12) 
    Custom = (True, ["operator"], ExpansionLevel.Vector, 13)

    def __init__(
            self,
            renormalize: bool,
            required_params: List[str],
            required_expansion_level: ExpansionLevel,
            op_id: int) -> None:
        self.renormalize = renormalize
        self.required_params = required_params
        self.required_expansion_level = required_expansion_level

    def update(self, **kwargs: Any) -> None:
        """
        Empty method, doesn't do anything in PolarizationOperationType
        """
        return

    def compute_operator(self, dimensions: List[int], **kwargs:Any) -> jnp.ndarray:
        """
        Computes an operator

        Parameters
        ----------
        dimensions: List[int]
            Accepts dimensions list, but it does not have any effect
        **kwargs: Any
            Accepts the kwargs, where the parameters are passed

        Returns
        -------
        jnp.ndarray
            Returns operator matrix
        """
        match self:
            case PolarizationOperationType.I:
                return identity_operator()
            case PolarizationOperationType.X:
                return x_operator()
            case PolarizationOperationType.Y:
                return y_operator()
            case PolarizationOperationType.Z:
                return z_operator()
            case PolarizationOperationType.H:
                return hadamard_operator()
            case PolarizationOperationType.S:
                return s_operator()
            case PolarizationOperationType.T:
                return t_operator()
            case PolarizationOperationType.SX:
                return sx_operator()
            case PolarizationOperationType.RX:
                return rx_operator(kwargs["theta"])
            case PolarizationOperationType.RY:
                return ry_operator(kwargs["theta"])
            case PolarizationOperationType.RZ:
                return rz_operator(kwargs["theta"])
            case PolarizationOperationType.U3:
                return u3_operator(kwargs["phi"], kwargs["theta"], kwargs["omega"])

    def compute_dimensions(self, num_quanta: int, state:jnp.ndarray, threshold:float = 1, **kwargs:Any) -> int:
        """
        Computes operation

        Parameters
        ----------
        num_quanta: int
            Accepts num quatna, but it doesn't have an effect
        state: jnp.ndarray
            Accepts traced out state of the state on which the operator
            will operate, doens't have an effect
        threshold: float
            Threshold value, doesn't have any effect
        **kwargs: Any
            Accepts parameters, but they do not have any effect

        Returns
        -------
        int
            Returns number of dimensions needed (always 2)
        """
        return 2
