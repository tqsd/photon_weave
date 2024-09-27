"""
Operations on Custom States
"""

from enum import Enum
import jax.numpy as jnp
from typing import List, Any, Union

from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.extra import interpreter


class CustomStateOperationType(Enum):
    """
    CustomStateOperationType

    Expression
    ----------
    Construcs an operator based on the expression provided.
    Alongside expression also a context needs to be provided.
    Context needs to be a dictionary of operators, where the keys are strings
    used in the expression and values are lambda functions, expecting one argument
    dimension. In this case the lambda can also not use the dims parameter,
    but it needs to be defined. Keep in mind that the resulting operator needs
    to have the same dimensionality of the state it is designed to operate on.

    An example of usage:
    >>> context = {
    >>>     "a": lambda dims: jnp.array([[0,0,0],[1,0,0],[0,0,0]]),
    >>>     "b": lambda dims: jnp.array([[0,0,0],[0,0,0],[0,1,0]])
    >>> }
    >>> expr = ("add", "a", "b")
    >>> op = Operation(CustomStateOperationType.Expresion, expr=expr, context=context)

    Custom
    ------
    Constructs a custom operator, which means that the oeprator needs to be manually
    provided. The dimensions of the operator must match the dimensions of the state,
    on which the operator is operating.

    An example of usage:
    >>> operator = jnp.array(
    >>>     [[0,0,0],
    >>>      [1,0,0],
    >>>      [0,1,0]]
    >>> )
    >>> op = Operation(CustomStateOperationType.Custom, operator=operator)
    """

    Expresion = (True, ["expr", "context"], ExpansionLevel.Vector, 1)
    Custom = (True, ["operator"], ExpansionLevel.Vector, 2)

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
        Empty method, doesn't do anything in FockOperationType
        """
        return

    def compute_operator(self, dimensions: List[int], **kwargs: Any) -> jnp.ndarray:
        """
        Generates the operator for this operation

        Parameters
        ----------
        dimensions: List[int]
            The dimensions of the sstate given in list, for this
            class only one integer should be given in list.
        **kwargs: Any
            Key word arguments for the operator creation
        """
        match self:
            case CustomStateOperationType.Custom:
                return kwargs["operator"]
            case CustomStateOperationType.Expresion:
                return interpreter(kwargs["expr"], kwargs["context"], dimensions)
        raise ValueError("Operator not recognized")

    def compute_dimensions(
        self,
        num_quanta: Union[int, List[int]],
        state: Union[jnp.ndarray, List[jnp.ndarray]],
        threshold: float = 1,
        **kwargs: Any,
    ) -> List[int]:
        """
        Compute the dimensions for the operator. Since this operator operates
        on a Custom State, which doesn't change dimensions, it always returns
        the dimensions of the given state

        Parameters
        ----------
        num_quanta: Union[int, List[int]]
            Does not have any effect on the output
        state: Union[jnp.ndarray, List[jnp.ndarray]]
            Traced out state, dimensions are computed from this state
        threshold: float
            Does not have any effect on the output
        **kwargs: Any
            Does not have any effect on the output

        Returns
        -------
        List[int]
            List of dimensions with only one element (state.shape[0])
        """
        assert isinstance(state, jnp.ndarray)
        return [state.shape[0]]
