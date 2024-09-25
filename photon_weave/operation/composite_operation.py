
from enum import Enum
from typing import Any, List
import jax.numpy as jnp
from jax.scipy.linalg import expm
import importlib


from photon_weave.extra import interpreter
from photon_weave._math.ops import (
    creation_operator,
    annihilation_operator
)
from photon_weave.extra import interpreter
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.base_state import BaseState
#from photon_weave.state.fock import Fock

class CompositeOperationType(Enum):
    """
    CompositeOperationType

    Constructs an operator, which acts on multiple spaces

    NonPolarizingBeamSplitter
    -------------------------
    Constructs a non-polarizing  beam splitter operator that acts on two Fock spaces.
    The operator is represented by a unitary transformation that mices the two modes.

    The operator transforms the annihilation and creation operators for two modes
    \( \hat{a} \) and \( \hat{b} \) as follows:

    \[
    \hat{a}' = \cos(\theta) \hat{a} + \sin(\theta) \hat{b}
    \]
    \[
    \hat{b}' = \sin(\theta) \hat{a} + \cos(\theta) \hat{b}
    \]

    For a 50/50 beam splitter , \( \theta = \frac{\pi}{4}\), which leas to equal
    mixing of the two modes.

    The full unitary \( \hat{U}_{BS} \) corresponding to this transformation is:
    \[
    \hat{U}_{BS} = \exp\left( i \eta \left( \hat{a}^\dagger \hat{b} + \hat{a} \hat{b}^\dagger \right) \right)
    \]

    Expression
    ----------
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
    >>> op = Operation(CompositeOperationType.Expression,
    >>>                expr=("expm"("s_mult", 1j,jnp.pi, "a"))),
    >>>                state_types=(Fock,), # Is applied to only one fock space
    >>>                context=context)
    >>> ce.apply_operation(op, fock_in_ce)

    It is IMPORTANT to correctly use the indexing in the context dictionary, when using dim[0],
    this operator will be used for the first state. And the dimensions of first state will be
    passed to the lambda function.

    It is also IMPORTANT to correctly write the expression as expression doesn't have a dimension
    asserting functionality, meaning that if you operate on two states, you need to correctly
    kron the operators to achieve correct operator dimensionality.

    For example if you which to operate on two Fock spaces like in beam splitter example you need
    to produce the following context and expression:
    >>> context = {
    >>>     'a_dag': lambda dims: creation_opreator(dims[0])
    >>>     'a':     lambda dims: annihilation_opreator(dims[0])
    >>>     'b_dag': lambda dims: creation_opreator(dims[1])
    >>>     'b':     lambda dims: annihilation_opreator(dims[1])
    >>> }
    >>> expression = ('expm',
    >>>                ('s_mult',
    >>>                  1j, jnp.pi/4,
    >>>                  ('add',
    >>>                    ('kron', a_dag, b),
    >>>                    ('kron', a, b_dag))))
    >>> op = Operation(
    >>>         CompositeOperationType.Expresion,
    >>>         expr=expression,
    >>>         state_types=(Fock, Fock),
    >>>         context=context)
    >>> ce.apply_operation(op, fock1, fock2)

    In this example 'a_dag' and 'a' will correspond to the fock1 state and
    'b_dag' and 'b' will correspond to the fock2 state. Notice also that the
    operators are always kronned in the order that corresponds to the usage later
    on. Application of the operation already reorders the states in a product state,
    such that their order corresponds to the order called in the apply_operation
    method call.
    """

    NonPolarizingBeamSplitter = (True, ["eta"], ["Fock", "Fock"], ExpansionLevel.Vector, 1)
    Expression = (True, ["expr","state_types", "context"], [], ExpansionLevel.Vector, 2)

    def __init__(self, renormalize: bool, required_params: list,
                 expected_base_state_types: List[BaseState],
                 required_expansion_level: ExpansionLevel,
                 op_id) -> None:
        
        self.renormalize = renormalize
        self.required_params = required_params
        self.expected_base_state_types = expected_base_state_types
        self.required_expansion_level = required_expansion_level

    def update(self, **kwargs: Any) -> None:
        """
        Updates the required_base_types in the case of Expression type

        Parameters
        ----------
        **kwargs: Any
            kwargs, where "state_types" is included
        """
        Fock = importlib.import_module('photon_weave.state.fock').Fock
        Polarization = importlib.import_module('photon_weave.state.polarization').Polarization
        CustomState = importlib.import_module('photon_weave.state.custom_state').CustomState
        if self is CompositeOperationType.Expression:
            self.expected_base_state_types = list(kwargs["state_types"])
        for i, state_type in enumerate(self.expected_base_state_types):
            if state_type == "Fock":
                self.expected_base_state_types[i] = Fock
            elif state_type == "Polarization":
                self.expected_base_state_types[i] = Polarization
            elif state_type == "CustomState":
                self.expected_base_state_types[i] = CustomState 

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
            case CompositeOperationType.Expression:
                return interpreter(kwargs["expr"], kwargs["context"], dimensions)
        
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
            case CompositeOperationType.Expression:
                return [d+1 for d in num_quanta]
