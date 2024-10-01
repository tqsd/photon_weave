from enum import Enum
from typing import Any, List, Tuple, Union

import jax.numpy as jnp

from photon_weave._math.ops import (
    hadamard_operator,
    identity_operator,
    rx_operator,
    ry_operator,
    rz_operator,
    s_operator,
    sx_operator,
    t_operator,
    u3_operator,
    x_operator,
    y_operator,
    z_operator,
)
from photon_weave.state.expansion_levels import ExpansionLevel


class PolarizationOperationType(Enum):
    r"""
    PolarizationOperationType

    Constructs an operator, which acts on a single Polarization Space

    Identity (I)
    ------------
    Constructs Identity (:math:`\hat{I}`) operator    
    .. math::

        \hat{I} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}

    Pauli X (X)
    -----------
    Constructs Pauli X  (:math:`\hat{\sigma_X}`) operator

    .. math::

        \hat{\sigma_X} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

    Pauli Y (Y)
    -----------
    Constructs Pauli Y  (:math:`\hat{\sigma_Y}`) operator

    .. math::

        \hat{\sigma_y} = \begin{bmatrix}
            0 & -i \\
            i & 0
            \end{bmatrix}

    Pauli Z (Z)
    -----------
    Constructs Pauli Z  (:math:`\hat {\sigma_Z}`) operator

    .. math::

        \hat{\sigma_Z} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}

    Hadamard (H)
    ------------
    Constructs Hadamard (:math:`\hat{H}`) operator

    .. math::

        \hat{H} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
    Phase Opeator (S)
    -----------------
    Constructs Phase (:math:`\hat S`) operator
    .. math::

        \hat{S} = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}

    T Opeator (T)
    -------------
    Constructs T (:math:`\hat T`) operator
    .. math::
        \hat{T} = \begin{bmatrix} 1 & 0 \\ 0 & e^{i \pi/4} \end{bmatrix}

    Sqrt(X) Operator (SX)
    ---------------------
    Constructs SX (:math:`\hat{SX}`) operator    
    .. math::

        \hat{SX} = \frac{1}{2} \begin{bmatrix} 1+i & 1-i \\ 1-i & 1+i \end{bmatrix}

    RX Operator (RX)
    ----------------
    Constructs RX (:math:`\hat{RX}(\theta)`) operator
    It rotates around X axis for given :math:`\theta` angle
    Requires an argument "theta" (:math:`\theta`)
    .. math::

        \hat{RX} = \begin{bmatrix}
        \cos\left(\frac{\theta}{2}\right) & -i\sin\left(\frac{\theta}{2}\right) \\
        -i\sin\left(\frac{\theta}{2}\right) & \cos\left(\frac{\theta}{2}\right)
        \end{bmatrix}

    Usage Example
    >>> op = Operation(PolarizationOperationType.RX, theta=jnp.pi)

    RY Operator (RY)
    ----------------
    Constructs RY (:math:`\hat{RY}(\theta)`) operator
    It rotates around Y axis for given :math:`\theta` angle
    Requires an argument "theta" (:math:`\theta`)

    .. math::

        \hat{RY} = \begin{bmatrix}
        \cos\left(\frac{\theta}{2}\right) & -\sin\left(\frac{\theta}{2}\right) \\
        \sin\left(\frac{\theta}{2}\right) & \cos\left(\frac{\theta}{2}\right)
        \end{bmatrix}

    Usage Example
    >>> op = Operation(PolarizationOperationType.RY, theta=jnp.pi)

    RZ Operator (RZ)
    ----------------
    Constructs RZ (:math:`\hat{RZ}(\theta)`) operator
    It rotates around Z axis for given :math:`\theta` angle
    Requires an argument "theta" (:math:`\theta`)    
    .. math::

        \hat{RZ} = \begin{bmatrix}
        e^{-i\frac{\theta}{2}} & 0 \\
        0 & e^{i\frac{\theta}{2}}
        \end{bmatrix}

    Usage Example
    >>> op = Operation(PolarizationOperationType.RZ, theta=jnp.pi)

    U3 operator
    -----------
    Rotation with 3 Euler angles (:math:`\hat{U3}(\phi,\theta,\omega))`)
    Requires three arguments "phi", "theta", "omega"

    Usage Example
    >>> op = Operation(PolarizationOperationType.RZ, theta=jnp.pi)

    Custom
    ------
    Constructs a custom operator, the operator needs to be given manually

    Usage Example
    >>> operator = jnp.array([[1,0],[0,1]])
    >>> op = Operation(PolarizationOperationType.Custom, operator=operator)
    """

    I: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        [],
        ExpansionLevel.Vector,
        1,
    )
    X: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        [],
        ExpansionLevel.Vector,
        2,
    )
    Y: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        [],
        ExpansionLevel.Vector,
        3,
    )
    Z: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        [],
        ExpansionLevel.Vector,
        4,
    )
    H: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        [],
        ExpansionLevel.Vector,
        5,
    )
    S: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        [],
        ExpansionLevel.Vector,
        6,
    )
    T: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        [],
        ExpansionLevel.Vector,
        7,
    )
    SX: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        [],
        ExpansionLevel.Vector,
        8,
    )
    RX: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        ["theta"],
        ExpansionLevel.Vector,
        9,
    )
    RY: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        ["theta"],
        ExpansionLevel.Vector,
        10,
    )
    RZ: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        ["theta"],
        ExpansionLevel.Vector,
        11,
    )
    U3: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        ["phi", "theta", "omega"],
        ExpansionLevel.Vector,
        12,
    )
    Custom: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        ["operator"],
        ExpansionLevel.Vector,
        13,
    )

    def __init__(
        self,
        renormalize: bool,
        required_params: List[str],
        required_expansion_level: ExpansionLevel,
        op_id: int,
    ) -> None:
        self.renormalize = renormalize
        self.required_params = required_params
        self.required_expansion_level = required_expansion_level

    def update(self, **kwargs: Any) -> None:
        """
        Empty method, doesn't do anything in PolarizationOperationType
        """
        return

    def compute_operator(self, dimensions: List[int], **kwargs: Any) -> jnp.ndarray:
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
            case PolarizationOperationType.Custom:
                return kwargs["operator"]
        raise ValueError("Operator not recognized")

    def compute_dimensions(
        self,
        num_quanta: Union[int, List[int]],
        state: Union[jnp.ndarray, List[jnp.ndarray]],
        threshold: float = 1,
        **kwargs: Any,
    ) -> List[int]:
        """
        Computes required dimensions for the operator. In this
        case the required dimension is always 2.

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
        return [2]
