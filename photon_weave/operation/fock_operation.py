"""
Operations on fock spaces
"""

from enum import Enum
from typing import Any, List, Tuple, Union

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
    r"""
    FockOperationType

    Constructs an operator, which acts on a single Fock space.

    Creation
    --------
    Constructs a creation operator :math:`\hat a^\dagger`
    Application to a fock state results in
    :math:`\hat a^\dagger |n\rangle = \sqrt{n+1}|n+1\rangle`
    The operator however normalizes the state afterwards, to when using
    Creation operator the resulting state is:
    :math:`\hat a^\dagger |n\rangle = |n+1\rangle`
    The dimensions of the Fock space is changed to the highest number state
    with non zero amplitude plus two.

    Annihilation
    ------------
    Constructs an annihlation operator :math:`\hat a`
    Application to a fock state results in
    :math:`\hat a|n\rangle = \sqrt{n}|n-1\rangle`
    The operator however normalizes the state afterwards, to when using
    Creation operator the resulting state is:
    :math:`\hat a|n\rangle = |n-1\rangle`
    The dimensions of the Fock space is changed to the highest number
    state with non zero amplitude plust two.

    PhaseShift
    ----------
    Constructs a phase shift operator :math:`\hat U(\phi)=e^{-i\phi\hat n}`
    The operation of this operator on a matrix state acts as a identity,
    since global shift is not representable in the density matrix.
    The dimension of the Fock space remains unchanged.

    Squeeze
    -------
    Constructs a Squeezing operator
    :math:`\hat S(\zeta)=e^{\frac{1}{2}(z^*\hat a^2 - z \hat a^\dagger^2)}`
    The dimensions of the Fock space required to accurately represent the application
    of this operator is iteratively determined, resulting in appropriate dimension.

    Displace
    --------
    Constructs a Squeezing operator
    :math:`\hat D(\alpha)=e^{\alpha \hat a^\dagger - \alpha^* \hat a}`
    The dimensions of the Fock space required to accurately represent the application
    of this operator is iteratively determined, resulting in appropriate dimension.
    The dimensions of the Fock space required to accurately represent the application
    of this operator is iteratively determined, resulting in appropriate dimension.

    Identity
    --------
    Constructs a Squeezing operator :math:`\hat I`
    This operator has no effect on the quantum state. The dimensions remain unchanged.

    Custom
    ------
    Constructs a custom operator, which means the operator needs to be manually
    provided, one must pay attention to the fact that operators dimensions need to match
    the dimensions of the Fock space. If the operator is larger than the underlying
    fock space, then the dimensions of the respective Fock space will increase to match
    the provided operator. Otherwise the state will be shrunk. If shrinking is not
    succesfull then the operation will fail. Shinking is not succesfull if part of the
    state is removed by the process of shrinking.

    Usage Example:
    >>> operator = jnp.ndarray(
    >>>    [[0,0],
    >>>     [0,1]]
    >>> )
    >>> op = Operation(FockOperationType.Custom, operator=operator)

    Expresion
    ---------
    Constucts an operator based on the expression provided. Alongside expression
    also list of state types needs to be provided together with the context.
    Context needs to be a dictionary of operators, where the keys are strings used in
    the expression and values are lambda functions, expecting one argument: dimensions.

    An example of context and operator usage
    >>> context = {
    >>>    "a_dag": lambda dims: creation_operator(dims[0])
    >>>    "a":     lambda dims: annihilation_operator(dims[0])
    >>>    "n":     lambda dims: number_operator(dims[0])
    >>> }
    >>> op = Operation(FockOperationType.Expression,
    >>>                expr=("expm"("s_mult", 1j,jnp.pi, "a"))),
    >>>                state_types=(Fock,), # Is applied to only one fock space
    >>>                context=context)
    >>> fock.apply_operation(op)
    """

    Creation: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        [],
        ExpansionLevel.Vector,
        1,
    )
    Annihilation: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        [],
        ExpansionLevel.Vector,
        2,
    )
    PhaseShift: Tuple[bool, List[str], ExpansionLevel, int] = (
        False,
        ["phi"],
        ExpansionLevel.Vector,
        3,
    )
    Squeeze: Tuple[bool, List[str], ExpansionLevel, int] = (
        True,
        ["zeta"],
        ExpansionLevel.Vector,
        4,
    )
    Displace: Tuple[bool, List[str], ExpansionLevel, int] = (
        False,
        ["alpha"],
        ExpansionLevel.Vector,
        5,
    )
    Identity: Tuple[bool, List[str], ExpansionLevel, int] = (
        False,
        [],
        ExpansionLevel.Vector,
        6,
    )
    Custom: Tuple[bool, List[str], ExpansionLevel, int] = (
        False,
        ["operator"],
        ExpansionLevel.Vector,
        7,
    )
    Expresion: Tuple[bool, List[str], ExpansionLevel, int] = (
        False,
        ["expr", "context"],
        ExpansionLevel.Vector,
        8,
    )

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

    def update(self, **kwargs: Any) -> None:
        """
        Empty method, doesnt do anything in FockOperationType
        """
        return

    def compute_operator(self, dimensions: List[int], **kwargs: Any) -> jnp.ndarray:
        """
        Generates the operator for this operation, given
        the dimensions

        Parameters
        ----------
        dimensions: List[int]
            The dimensions of the state given in list, for
            this class only one element should be given in list.
            The reason for the list is so that the signature is
            same as in CompositeOperationType
        **kwargs: Any
            List of key word arguments for specific operator types
        """
        match self:
            case FockOperationType.Creation:
                return creation_operator(dimensions[0])
            case FockOperationType.Annihilation:
                return annihilation_operator(dimensions[0])
            case FockOperationType.PhaseShift:
                return phase_operator(dimensions[0], kwargs["phi"])
            case FockOperationType.Displace:
                return displacement_operator(dimensions[0], kwargs["alpha"])
            case FockOperationType.Squeeze:
                return squeezing_operator(dimensions[0], kwargs["zeta"])
            case FockOperationType.Identity:
                return jnp.identity(dimensions[0])
            case FockOperationType.Expresion:
                return interpreter(kwargs["expr"], kwargs["context"], dimensions)
            case FockOperationType.Custom:
                return kwargs["operator"]
        raise ValueError("Something went wrong in operation generation")

    def compute_dimensions(
        self,
        num_quanta: Union[int, List[int]],
        state: Union[jnp.ndarray, List[jnp.ndarray]],
        threshold: float = 1 - 1e-6,
        **kwargs: Any,
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
        List[int]
           New number of dimensions in a list for compatibility

        Notes
        -----
        This functionality is called before the operator is computed, so that
        the dimensionality of the space can be changed before the application
        and the dimensionality of the operator and space match
        """
        from photon_weave.operation.operation import Operation

        assert isinstance(num_quanta, int)
        assert isinstance(state, jnp.ndarray)

        match self:
            case FockOperationType.Creation:
                return [int(num_quanta + 2)]
            case FockOperationType.Annihilation:
                return [num_quanta + 2]
            case FockOperationType.PhaseShift:
                return [num_quanta + 1]
            case FockOperationType.Displace:
                fd = FockDimensions(
                    state,
                    Operation(FockOperationType.Displace, **kwargs),
                    num_quanta,
                    threshold,
                )
                return [fd.compute_dimensions()]
            case FockOperationType.Squeeze:
                fd = FockDimensions(
                    state,
                    Operation(FockOperationType.Squeeze, **kwargs),
                    num_quanta,
                    threshold,
                )
                return [fd.compute_dimensions()]
            case FockOperationType.Identity:
                return [num_quanta + 1]
            case FockOperationType.Expresion:
                fd = FockDimensions(
                    state,
                    Operation(FockOperationType.Expresion, **kwargs),
                    num_quanta,
                    threshold,
                )
                return [fd.compute_dimensions()]
        raise ValueError("Something went wrong in dimension estimation")
