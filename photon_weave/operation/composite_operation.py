import importlib
from enum import Enum
from typing import Any, List, Sequence, Union, Type, TYPE_CHECKING

import jax.numpy as jnp
from jax.scipy.linalg import expm

from photon_weave._math.ops import (
    annihilation_operator,
    controlled_not_operator,
    controlled_swap_operator,
    controlled_z_operator,
    creation_operator,
    swap_operator,
)

from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.extra import interpreter

if TYPE_CHECKING:
    from photon_weave.state.base_state import BaseState


class CompositeOperationType(Enum):
    r"""
    CompositeOperationType

    Constructs an operator, which acts on multiple spaces

    NonPolarizingBeamSplitter
    -------------------------
    Constructs a non-polarizing  beam splitter operator that acts on two Fock
    spaces. The operator is represented by a unitary transformation that mixes
    the two modes. The constructed operator is of the form:
    .. math::

        \hat U_{BS} = e^{i\theta(\hat a^\dagger \hat b +
        \hat a \hat b^\dagger)}

    For a 50/50 beam splitter, :math:`\theta = \frac{\pi}{4}\`, which leas to
    equal mixing of the two modes.

    CNOT (CXPolarization)
    -----------------------
    Constructs a CNOT operator operating on two polarization states. First
    state provided is control and the second is target

    .. math::

        \hat{CX} =
        \begin{bmatrix}
          1&0&0&0\\
          0&0&0&1\\
          0&0&1&0\\
          0&1&0&0
        \end{bmatrix}

    >>> op = Operation(CompositeOperationType.CXPolarization)
    >>> ce.apply_operation(op, control_pol, target_pol)

    CZ (CZPolarization)
    -----------------------
    Constructs a controlled-Z opreator operating on two polarization states.
    First state provided is control and the second is target.
    .. math::

        \hat{CZ} =
        \begin{bmatrix}
            1&0&0&0\\
            0&1&0&0\\
            0&0&1&0\\
            0&0&0&-1
        \end{bmatrix}

    >>> op = Operation(CompositeOperationType.CZPolarization)
    >>> ce.apply_operation(op, control_pol, target_pol)

    SWAP (SwapPolarization)
    -----------------------
    Constructs a SWAP operation, swaping the states of two provided
    polarization states.

    .. math::
        \hat{SWAP} =
        \begin{bmatrix}
            1&0&0&0\\
            0&0&1&0\\
            0&1&0&0\\
            0&0&0&1
        \end{bmatrix}

    Example Usage:
    >>> op = Operation(CompositeOperationType.SwapPolarization)
    >>> ce.apply_operation(op, env1.polarization, env2.polarization)

    Controlled-SWAP (CSwapPolarization)
    -----------------------------------
    Constructs a Controlled-SWAP operation, conditionally swapping the states
    of two provided polarization states. First state is the control state and
    the next two states are target states.
    .. math::

        \hat{CSWAP} =
        \begin{bmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}

    Example Usage:
    >>> op = Operation(CompositeOperationType.SwapPolarization)
    >>> ce.apply_operation(op, env1.polarization, env2.polarization)

    Expression
    ----------
    Constructs an operator based on the expression provided. Alongside
    expression also list of state types needs to be provided together with the
    context. Context needs to be a dictionary of operators, where the keys are
    strings used in the expression and values are lambda functions, expecting
    one argument: dimensions.

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

    It is IMPORTANT to correctly use the indexing in the context dictionary,
    when using dim[0], this operator will be used for the first state. And the
    dimensions of first state will be passed to the lambda function.

    It is also IMPORTANT to correctly write the expression as expression
    doesn't have a dimension asserting functionality, meaning that if you
    operate on two states, you need to correctly kron the operators to achieve
    correct operator dimensionality.

    For example if you which to operate on two Fock spaces like in
    beam-splitter example you need to produce the following context and
    expression:
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
    operators are always kronned in the order that corresponds to the usage
    later on. Application of the operation already reorders the states in a
    product state, such that their order corresponds to the order called in
    the apply_operation method call.
    """

    NonPolarizingBeamSplitter = (
        True,
        ["eta"],
        ["Fock", "Fock"],
        ExpansionLevel.Vector,
        1,
    )
    CXPolarization = (
        True,
        [],
        ["Polarization", "Polarization"],
        ExpansionLevel.Vector,
        2,
    )
    SwapPolarization = (
        True,
        [],
        ["Polarization", "Polarization"],
        ExpansionLevel.Vector,
        3,
    )
    CSwapPolarization = (
        True,
        [],
        ["Polarization" for _ in range(3)],
        ExpansionLevel.Vector,
        4,
    )
    CZPolarization = (
        True,
        [],
        ["Polarization" for _ in range(2)],
        ExpansionLevel.Vector,
        5,
    )
    Expression = (
        True,
        ["expr", "state_types", "context"],
        [],
        ExpansionLevel.Vector,
        6,
    )

    renormalize: bool
    required_params: List[str]
    expected_base_state_types: List[Type["BaseState"]]
    required_expansion_level: ExpansionLevel

    def __new__(
        cls,
        renormalize: bool,
        required_params: List[str],
        expected_base_state_types: Sequence[Union[Type["BaseState"], str]],
        expansion_level: ExpansionLevel,
        op_id: int,
    ):
        obj = object.__new__(cls)
        obj._value_ = op_id

        # Lazy resolve string class names to actual classes
        resolved_types: List[Type["BaseState"]] = []
        if expected_base_state_types:
            for st in expected_base_state_types:
                if isinstance(st, str):
                    mod_pol = importlib.import_module(
                        "photon_weave.state.polarization"
                    )
                    if st == "Polarization":
                        resolved_types.append(getattr(mod_pol, "Polarization"))
                    elif st == "Fock":
                        mod_fock = importlib.import_module(
                            "photon_weave.state.fock"
                        )
                        resolved_types.append(getattr(mod_fock, "Fock"))
                    elif st == "CustomState":
                        mod_custom = importlib.import_module(
                            "photon_weave.state.custom_state"
                        )
                        resolved_types.append(
                            getattr(mod_custom, "CustomState")
                        )
                    else:
                        raise ValueError(f"Unknown state type: {st}")
                else:
                    resolved_types.append(st)

        obj.expected_base_state_types = resolved_types
        obj.renormalize = renormalize
        obj.required_params = required_params
        obj.required_expansion_level = expansion_level
        return obj

    def update(self, **kwargs: Any) -> None:
        if (
            self is CompositeOperationType.Expression
            and "state_types" in kwargs
        ):
            from photon_weave.state.fock import Fock
            from photon_weave.state.polarization import Polarization
            from photon_weave.state.custom_state import CustomState

            resolved = []
            for st in kwargs["state_types"]:
                if st == "Fock":
                    resolved.append(Fock)
                elif st == "Polarization":
                    resolved.append(Polarization)
                elif st == "CustomState":
                    resolved.append(CustomState)
                else:
                    resolved.append(st)  # already a class
            self.expected_base_state_types = resolved

    def compute_operator(
        self, dimensions: List[int], **kwargs: Any
    ) -> jnp.ndarray:
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

                operator = jnp.kron(a_dagger, b) + jnp.kron(a, b_dagger)
                return expm(1j * kwargs["eta"] * operator)
            case CompositeOperationType.CXPolarization:
                return controlled_not_operator()
            case CompositeOperationType.SwapPolarization:
                return swap_operator()
            case CompositeOperationType.CSwapPolarization:
                return controlled_swap_operator()
            case CompositeOperationType.CZPolarization:
                return controlled_z_operator()
            case CompositeOperationType.Expression:
                return interpreter(
                    kwargs["expr"], kwargs["context"], dimensions
                )
        raise ValueError("Operation Type not recognized")

    def compute_dimensions(
        self,
        num_quanta: Union[int, List[int]],
        states: Union[jnp.ndarray, List[jnp.ndarray]],
        threshold: float = 1 - 1e6,
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
        int
           New number of dimensions

        Notes
        -----
        This functionality is called before the operator is computed, so that
        the dimensionality of the space can be changed before the application
        and the dimensionality of the operator and space match
        """
        from photon_weave.state.fock import Fock

        match self:
            case CompositeOperationType.NonPolarizingBeamSplitter:
                dim = int(jnp.sum(jnp.array(num_quanta))) + 1
                return [dim, dim]
            case CompositeOperationType.CXPolarization:
                return [2, 2]
            case CompositeOperationType.SwapPolarization:
                return [2, 2]
            case CompositeOperationType.CSwapPolarization:
                return [2, 2, 2, 2]
            case CompositeOperationType.CZPolarization:
                return [2, 2]
            case CompositeOperationType.Expression:
                assert isinstance(num_quanta, list)
                dims = []
                for i, s in enumerate(self.expected_base_state_types):
                    if s is Fock:
                        dims.append(num_quanta[i] + 2)
                    else:
                        dims.append(num_quanta[i])
                return dims
        raise ValueError("Operation Type not recognized")
