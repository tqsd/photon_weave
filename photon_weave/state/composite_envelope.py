from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, cast

import jax.numpy as jnp
import opt_einsum as oe

# from photon_weave.extra.einsum_constructor import EinsumStringConstructor as ESC
import photon_weave.extra.einsum_constructor as ESC
from photon_weave.core.ops import (
    kraus_identity_check,
    kron_reduce,
    num_quanta_matrix,
    num_quanta_vector,
)
from photon_weave.core.rng import borrow_key
from photon_weave.operation import (
    CompositeOperationType,
    CustomStateOperationType,
    FockOperationType,
    Operation,
    PolarizationOperationType,
)
from photon_weave.photon_weave import Config
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.interfaces import (
    BaseStateLike as BaseState,
)
from photon_weave.state.interfaces import (
    EnvelopeLike as Envelope,
)

from .utils import shape_planning
from .utils.measurements import (
    measure_matrix_expectation,
    measure_matrix_jit,
    measure_POVM_matrix,
    measure_vector_expectation,
    measure_vector_jit,
)
from .utils.operations import apply_kraus_matrix
from .utils.shape_planning import ShapePlan, compiled_kernels
from .utils.state_transform import (
    state_contract,
    state_expand,
    state_expand_jit,
)
from .utils.trace_out import trace_out_matrix, trace_out_vector


def _is_envelope(obj: object) -> bool:
    return hasattr(obj, "fock") and hasattr(obj, "polarization")


def _is_custom_state(obj: object) -> bool:
    return obj.__class__.__name__ == "CustomState"


def _is_fock(obj: object) -> bool:
    return obj.__class__.__name__ == "Fock"


def _is_polarization(obj: object) -> bool:
    return obj.__class__.__name__ == "Polarization"


class ProductState:
    """
    Stores Product state and references to its constituents
    """

    __slots__ = (
        "expansion_level",
        "container",
        "uid",
        "state",
        "state_objs",
        "_meta_cache",
        "_meta_key",
        "_plan_cache",
    )

    expansion_level: ExpansionLevel
    container: CompositeEnvelopeContainer
    uid: uuid.UUID
    state: jnp.ndarray
    state_objs: List[BaseState]
    _meta_cache: shape_planning.DimsMeta | None
    _plan_cache: Dict[
        Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], ShapePlan
    ]

    def __init__(
        self,
        expansion_level: ExpansionLevel,
        container: CompositeEnvelopeContainer,
        state: Optional[jnp.ndarray] = None,
        state_objs: Optional[List[BaseState]] = None,
    ) -> None:
        self.expansion_level = expansion_level
        self.container = container
        self.uid = uuid.uuid4()
        self.state = jnp.array([[1]]) if state is None else state
        self.state_objs = [] if state_objs is None else state_objs
        self._meta_cache = None
        self._meta_key: Tuple[Tuple[int, ...], Tuple[int, ...]] | None = None
        self._plan_cache = {}

    def __hash__(self) -> int:
        return hash(self.uid)

    def expand(self) -> None:
        """
        Expands the state from vector to matrix
        """
        if self.expansion_level < ExpansionLevel.Matrix:
            dims = int(
                jnp.prod(jnp.array([s.dimensions for s in self.state_objs]))
            )
            if Config().use_jit:
                self.state, self.expansion_level = state_expand_jit(
                    self.state, self.expansion_level, dims
                )
            else:
                self.state, self.expansion_level = state_expand(
                    self.state, self.expansion_level, dims
                )
            for state in self.state_objs:
                state.expansion_level = ExpansionLevel.Matrix

    def contract(self, tol: float = 1e-6) -> None:
        """
        Attempts to contract the representation from matrix to vector

        Parameters
        ----------
        tol: float
            tolerance when comparing trace
        """
        if self.expansion_level is ExpansionLevel.Matrix:
            self.state, self.expansion_level, success = cast(
                tuple[jnp.ndarray, ExpansionLevel, bool],
                state_contract(self.state, self.expansion_level),
            )
            if success:
                for state in self.state_objs:
                    state.expansion_level = ExpansionLevel.Vector

    def prepare_meta(
        self, *target_states: BaseState
    ) -> shape_planning.DimsMeta:
        """
        Prepare static dimension metadata for this product state.

        If no targets are provided, all state_objs are targets.
        """
        targets: Tuple[BaseState, ...] = (
            tuple(target_states) if target_states else tuple(self.state_objs)
        )
        dims_key = tuple(s.dimensions for s in self.state_objs)
        meta_key = (dims_key, tuple(id(t) for t in targets))
        if self._meta_cache is None or self._meta_key != meta_key:
            self._meta_cache = shape_planning.build_meta(
                self.state_objs, targets
            )
            self._meta_key = meta_key
        assert self._meta_cache is not None
        return self._meta_cache

    def prepare_plan(self, *target_states: BaseState) -> ShapePlan:
        """
        Prepare a ShapePlan for this composite product state.
        """
        targets: Tuple[BaseState, ...] = (
            tuple(target_states) if target_states else tuple(self.state_objs)
        )
        plan_key = (
            tuple(s.dimensions for s in self.state_objs),
            tuple(id(s) for s in self.state_objs),
            tuple(id(t) for t in targets),
        )
        if plan_key not in self._plan_cache:
            self._plan_cache[plan_key] = shape_planning.build_plan(
                self.state_objs, targets
            )
        return self._plan_cache[plan_key]

    def compiled_kernels(self, *target_states: BaseState):
        """
        Return meta-bound, jitted kernels for this product state.

        If no target states are provided, all state_objs are targets.
        """
        plan = self.prepare_plan(*target_states)
        return compiled_kernels(plan)

    @property
    def size(self) -> int:
        """
        Returns the size of the product state

        Returns
        -------
        int
            The size of the product state in bytes
        """
        return self.state.nbytes

    def reorder(self, *ordered_states: "BaseState") -> None:
        """
        Changes the order of tensoring, all ordered states need to be given

        Parameters
        ----------
        *ordered_states: 'BaseState'
            States ordered in the new order
        """

        assert all(
            so in ordered_states for so in self.state_objs
        ), "All state objects need to be given"
        if self.expansion_level == ExpansionLevel.Vector:
            # Get the state and reshape it
            shape = [so.dimensions for so in self.state_objs]
            shape.append(1)
            state = self.state.reshape(shape)

            # Generate the Einsum String
            einsum = ESC.reorder_vector(self.state_objs, list(ordered_states))

            # Perform the reordering
            state = oe.contract(einsum, state, backend="jax")

            # Reshape and store the state
            self.state = state.reshape(-1, 1)

            # Update the new order
            self.state_objs = list(ordered_states)
        elif self.expansion_level == ExpansionLevel.Matrix:
            # Get the state and reshape it
            shape = [os.dimensions for os in self.state_objs] * 2
            state = self.state.reshape(shape)

            # Get the einstein sum string
            einsum = ESC.reorder_matrix(self.state_objs, list(ordered_states))

            # Perform reordering
            state = oe.contract(einsum, state, backend="jax")

            # Reshape and reorder self.state_objs to reflect the new order
            new_dims = jnp.prod(
                jnp.array([s.dimensions for s in ordered_states])
            )
            self.state = state.reshape((new_dims, new_dims))
            self.state_objs = list(ordered_states)

        # Update indices in all of the states in this product space
        self.container.update_all_indices()

    def measure(
        self,
        *states: "BaseState",
        separate_measurement: bool = False,
        destructive: bool = True,
        key: jnp.ndarray | None = None,
        return_key: bool = False,
    ) -> Tuple[Dict["BaseState", int], jnp.ndarray | None]:
        """
        Measures this subspace. If the state is measured partially, then the state
        are moved to their respective spaces. If the measurement is destructive, then
        the state is destroyed post measurement.

        Parameters
        ----------
        states: Optional[BaseState]
            Optional, when measuring spaces individualy
        separate_measurement:bool
            if True given states will be measured separately and the state which is not
            measured will be preserved (False by default)
        destructive: bool
            If False, the measurement will not destroy the state after the measurement.
            The state will still be affected by the measurement (True by default)

        Returns
        -------
        Dict[BaseState,int]
            Dictionary of outcomes, where the state is key and its outcome measurement
        is the value (int)
        jnp.ndarray or None
            Next key after measurement (if provided and `return_key` is True).
        """
        if key is None:
            raise ValueError(
                "PRNG key is required when measuring a composite product state"
            )
        assert all(
            so in self.state_objs for so in states
        ), "All state objects need to be in product state"

        plan = self.prepare_plan(*states)
        use_key, _ = borrow_key(key)
        match self.expansion_level:
            case ExpansionLevel.Vector:
                outcomes, self.state, next_key = measure_vector_jit(
                    self.state_objs,
                    states,
                    self.state,
                    use_key,
                    meta=plan,
                )
            case ExpansionLevel.Matrix:
                outcomes, self.state, next_key = measure_matrix_jit(
                    self.state_objs,
                    states,
                    self.state,
                    use_key,
                    meta=plan,
                )
        # Handle post measurement processes
        for state in states:
            if destructive:
                state._set_measured()
            else:
                if _is_polarization(state):
                    label_cls = getattr(state.state, "__class__", None)
                    if label_cls is not None and hasattr(label_cls, "H"):
                        state.state = (
                            label_cls.H
                            if outcomes[state] == 0
                            else label_cls.V
                        )
                    else:
                        state.state = outcomes[state]
                else:
                    state.state = outcomes[state]
                state.expansion_level = ExpansionLevel.Label
                state.index = None

        self.state_objs = [s for s in self.state_objs if s not in states]
        C = Config()
        if C.contractions and len(self.state_objs) > 0:
            self.contract()
        if return_key:
            return outcomes, next_key
        return outcomes

    def measure_expectation(
        self, *states: "BaseState"
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Differentiable expectation of measuring the given subsystems.
        Returns (probabilities, expected post-measurement density matrix).
        """
        assert all(
            so in self.state_objs for so in states
        ), "All state objects need to be in product state"
        plan = self.prepare_plan(*states)
        match self.expansion_level:
            case ExpansionLevel.Vector:
                return measure_vector_expectation(
                    self.state_objs, states, self.state, meta=plan
                )
            case ExpansionLevel.Matrix:
                return measure_matrix_expectation(
                    self.state_objs, states, self.state, meta=plan
                )
        raise ValueError(
            "Unsupported expansion level for expectation measurement"
        )

    def measure_POVM(
        self,
        operators: List[jnp.ndarray],
        *states: "BaseState",
        destructive: bool = True,
        key: jnp.ndarray | None = None,
    ) -> Tuple[int, Dict["BaseState", int]]:
        """
        Perform a POVM measurement.

        Parameters
        ----------
        operators: List[jnp.ndarray]
            List of POVM operators
        *stapartial=partials: BaseState
            List of states, on which the POVM measurement should be executed
            The order of the states must reflect the order of tensoring of
            individual hilbert space in the individual operator
        destructive: bool
            If desctructive is set to True, then the states will be
            destroyed post measurement

        Returns
        -------
        Tuple[int, Dict[Union['Fock', 'Polarization'], int]]
            Tuple, where the first element is the outcome of the POVM measurement
            and the other is a dictionary of outcomes if the measurement
            was desctructive and some states are not captured in the
            POVM operator measurements.
        """

        # Expand to matrix form if not already in matrix form
        if self.expansion_level == ExpansionLevel.Vector:
            self.expand()

        plan = self.prepare_plan(*states)
        outcome, self.state = measure_POVM_matrix(
            self.state_objs,
            states,
            operators,
            self.state,
            key,
            meta=plan.meta,
        )

        other_outcomes = {}
        if destructive:
            # Custom State cannot be destroyed
            for s in states:
                if _is_custom_state(s):
                    s.state = trace_out_matrix(
                        self.state_objs, [s], self.state
                    )
                    s.expansion_level = ExpansionLevel.Matrix
                    s.index = None

            remaining_states = [s for s in self.state_objs if s not in states]
            if isinstance(
                CompositeEnvelope._instances[self.container.composite_uid],
                list,
            ):
                other_outcomes = CompositeEnvelope._instances[
                    self.container.composite_uid
                ][0].measure(*states, key=key)
                for s in states:
                    del other_outcomes[s]
            self.state_objs = remaining_states

        C = Config()
        if C.contractions and len(self.state_objs) > 0:
            self.contract()
        return (outcome, other_outcomes)

    def apply_kraus(
        self,
        operators: Union[List[jnp.ndarray], Tuple[jnp.ndarray, ...]],
        *states: "BaseState",
    ) -> None:
        """
        Applies the Kraus oeprators to the selected states, called by the apply_kraus
        method in CopositeEnvelope

        Parameters
        ----------
        operators: operators:List[Union[np.ndarray, jnp.ndarray]]
            List of operators to be applied, must be tensored using kron
        *states:BaseState
            List of states to apply the operators to, the tensoring order in operators
            must follow the order of the states in this list
        """
        while self.expansion_level < ExpansionLevel.Matrix:
            self.expand()
        self.state = apply_kraus_matrix(
            self.state_objs,
            states,
            self.state,
            operators,
            use_contraction=Config().contractions,
        )
        C = Config()
        if C.contractions:
            self.contract()

    def trace_out(self, *states: "BaseState") -> jnp.ndarray:
        """
        Traces out the rest of the states from the product state and returns
        the resultint matrix or vector. If given states are in sparate product
        spaces it merges the product spaces.

        Parameters
        ----------
        *states: BaseState

        Returns
        -------
        jnp.ndarray
            Traced out system including only the requested states in tesored
            in the order in which the states are given
        """
        match self.expansion_level:
            case ExpansionLevel.Vector:
                return trace_out_vector(
                    self.state_objs, list(states), self.state
                )
            case ExpansionLevel.Matrix:
                return trace_out_matrix(
                    self.state_objs, list(states), self.state
                )

        return jnp.ndarray([[1]])

    @property
    def is_empty(self) -> bool:
        """
        Returns True if the product state is empty (it contains no sates)

        Returns
        -------
        bool: True if empty, False if not
        """
        if len(self.state_objs) == 0:
            return True
        return False

    def resize_fock(self, new_dimensions: int, fock: BaseState) -> bool:
        """
        Resizes the space to the new dimensions.
        If the dimensions are more, than the current dimensions, then
        it gets padded. If the dimensions are less then the current
        dimensions, then it checks if it can shrink the space.

        Parameters
        ----------
        new_dimensions: bool
            New dimensions to be set
        fock: Fock
            The fock state, for which the dimensions
            must be changed

        Returns
        -------
        bool
            True if the resizing was succesfull
        """
        if self.expansion_level == ExpansionLevel.Vector:
            shape = [so.dimensions for so in self.state_objs]
            shape.append(1)
            ps = self.state.reshape(shape)
            if new_dimensions > fock.dimensions:
                padding = new_dimensions - fock.dimensions
                pad_config = [(0, 0) for _ in range(ps.ndim)]
                assert isinstance(fock.index, tuple)
                pad_config[fock.index[1]] = (0, padding)
                ps = jnp.pad(
                    ps, pad_config, mode="constant", constant_values=0
                )
                fock.dimensions = new_dimensions
                dims = jnp.prod(
                    jnp.array([s.dimensions for s in self.state_objs])
                )
                self.state = ps.reshape((dims, 1))
                return True
            if new_dimensions < fock.dimensions:
                to = fock.trace_out()
                assert isinstance(to, jnp.ndarray)
                assert isinstance(fock.index, tuple)
                num_quanta = num_quanta_vector(to)
                if num_quanta >= new_dimensions:
                    return False
                slices = [slice(None)] * ps.ndim
                slices[fock.index[1]] = slice(0, new_dimensions)
                ps = ps[tuple(slices)]
                fock.dimensions = new_dimensions
                dims = jnp.prod(
                    jnp.array([s.dimensions for s in self.state_objs])
                )
                self.state = ps.reshape((dims, 1))
                return True

        if self.expansion_level == ExpansionLevel.Matrix:
            shape = [so.dimensions for so in self.state_objs]
            transpose_pattern = [
                item
                for i in range(len(self.state_objs))
                for item in [i, i + len(self.state_objs)]
            ]
            ps = self.state.reshape([*shape, *shape]).transpose(
                transpose_pattern
            )
            if new_dimensions > fock.dimensions:
                assert isinstance(fock.index, tuple)
                padding = new_dimensions - fock.dimensions
                pad_config = [(0, 0) for _ in range(ps.ndim)]
                pad_config[fock.index[1] * len(self.state_objs)] = (0, padding)
                pad_config[fock.index[1] * len(self.state_objs) + 1] = (
                    0,
                    padding,
                )
                ps = jnp.pad(
                    ps, pad_config, mode="constant", constant_values=0
                )
                fock.dimensions = new_dimensions
                dims = jnp.prod(
                    jnp.array([s.dimensions for s in self.state_objs])
                )
                ps = ps.transpose(transpose_pattern)
                self.state = ps.reshape((dims, dims))
                return True
            if new_dimensions < fock.dimensions:
                to = fock.trace_out()
                assert isinstance(to, jnp.ndarray)
                assert isinstance(fock.index, tuple)
                num_quanta = num_quanta_matrix(to)
                if num_quanta >= new_dimensions:
                    return False
                slices = [slice(None)] * ps.ndim
                slices[fock.index[1] * len(self.state_objs)] = slice(
                    0, new_dimensions
                )
                slices[fock.index[1] * len(self.state_objs) + 1] = slice(
                    0, new_dimensions
                )
                ps = ps[tuple(slices)]
                fock.dimensions = new_dimensions
                ps = ps.transpose(transpose_pattern)
                dims = jnp.prod(
                    jnp.array([s.dimensions for s in self.state_objs])
                )
                self.state = jnp.array(ps.reshape((dims, dims)))
                return True
        return False

    def apply_operation(
        self, operation: Operation, *states: "BaseState"
    ) -> None:
        """
        Apply operation to the given states in this product state

        Parameters
        ----------
        operation: Operation
            The operation which will be applied
        *states: BaseState
            The states in the correct order to which the operation
            will be applied
        """
        if not jnp.any(jnp.abs(self.state) > 0):
            raise ValueError("The state is invalid" "State 0")

        C = Config()
        if C.use_jit and C.dynamic_dimensions:
            raise ValueError(
                "JIT mode does not support dynamic dimension resizing; "
                "pre-size states or disable use_jit."
            )
        if isinstance(operation._operation_type, FockOperationType):
            if not _is_fock(states[0]):
                raise ValueError("Fock operation requires a Fock state")
            assert len(states) == 1
            if C.dynamic_dimensions:
                to = states[0].trace_out()
                assert isinstance(to, jnp.ndarray)
                operation.compute_dimensions(states[0]._num_quanta, to)
                target_dim = operation.dimensions[0]
                if C.use_jit and target_dim != states[0].dimensions:
                    raise ValueError(
                        "JIT mode requires static Fock dimensions; "
                        f"expected {states[0].dimensions}, got {target_dim}. "
                        "Pre-size the state or disable use_jit."
                    )
                states[0].resize(target_dim)
            else:
                operation.dimensions = [states[0].dimensions]

        elif isinstance(operation._operation_type, PolarizationOperationType):
            if not _is_polarization(states[0]):
                raise ValueError(
                    "Polarization operation requires a Polarization state"
                )
            assert len(states) == 1
            # Parameters doesn't have any effect
            operation.compute_dimensions(0, jnp.array([0]))
        elif isinstance(operation._operation_type, CustomStateOperationType):
            if not _is_custom_state(states[0]):
                raise ValueError(
                    "CustomState operation requires a CustomState"
                )
            assert len(states) == 1
            to = states[0].trace_out()
            assert isinstance(to, jnp.ndarray)
            operation.compute_dimensions(0, to)
        elif isinstance(operation._operation_type, CompositeOperationType):
            assert len(states) == len(
                operation._operation_type.expected_base_state_types
            )
            for i, s in enumerate(states):
                op_type = operation._operation_type
                assert isinstance(
                    s, op_type.expected_base_state_types[i]  # type: ignore
                )
            if C.use_jit and C.dynamic_dimensions:
                raise ValueError(
                    "JIT mode does not support dynamic dimension resizing; "
                    "pre-size states or disable use_jit."
                )
            if C.dynamic_dimensions:
                operation.compute_dimensions(
                    [
                        s._num_quanta if _is_fock(s) else s.dimensions
                        for s in states
                    ],
                    [s.trace_out() for s in states],  # type: ignore
                )
                for i, s in enumerate(states):
                    if _is_fock(s):
                        s.resize(operation._dimensions[i])
            else:
                operation.dimensions = [s.dimensions for s in states]

        # plan = self.prepare_plan(*states)
        kernels = self.compiled_kernels(*states)
        match self.expansion_level:
            case ExpansionLevel.Vector:
                self.state = kernels.apply_op_vector(
                    self.state,
                    operation.operator,
                    use_contraction=C.contractions,
                )
            case ExpansionLevel.Matrix:
                self.state = kernels.apply_op_matrix(
                    self.state,
                    operation.operator,
                    use_contraction=C.contractions,
                )
        if not jnp.any(jnp.abs(self.state) > 0):
            raise ValueError(
                "The state is entirely composed of zeros, "
                "is |0⟩ attempted to be annihilated?"
            )
        if operation.renormalize:
            self.state = self.state / jnp.linalg.norm(self.state)

        # Reduce unneeded dimensionality of Fock spaces
        for so in self.state_objs:
            if _is_fock(so):
                match so.expansion_level:
                    case ExpansionLevel.Vector:
                        num_quanta = max(1, num_quanta_vector(so.trace_out()))
                    case ExpansionLevel.Matrix:
                        num_quanta = max(1, num_quanta_matrix(so.trace_out()))
                so.resize(num_quanta + 1)
        C = Config()
        if C.contractions:
            self.contract()


@dataclass(slots=True)
class CompositeEnvelopeContainer:
    composite_uid: uuid.UUID
    envelopes: List["Envelope"] = field(default_factory=list)
    state_objs: List["BaseState"] = field(default_factory=list)
    states: List[ProductState] = field(default_factory=list)

    def append_states(self, other: "CompositeEnvelopeContainer") -> None:
        """
        Appends the states of two composite envelope containers
        Parameters
        ----------
        other: CompositeEnvelopeContainer
            Other composite envelope container
        """
        assert isinstance(other, CompositeEnvelopeContainer)
        self.states.extend(other.states)
        self.envelopes.extend(other.envelopes)

    def remove_empty_product_states(self) -> None:
        """
        Checks if a product state is empty and if so
        removes it
        """
        for state in self.states[:]:
            if state is not None:
                if state.is_empty:
                    self.states.remove(state)

    def update_all_indices(self) -> None:
        """
        Updates all of the indices of the state_objs
        """
        for state_index, state in enumerate(self.states):
            for i, so in enumerate(state.state_objs):
                if so is not None:
                    so.extract((state_index, i))
                    so.composite_envelope = CompositeEnvelope._instances[
                        self.composite_uid
                    ][0]


class CompositeEnvelope:
    """
    Composite Envelope is a pointer to a container, which includes the state
    Multiple Composite envelopes can point to the same containers.
    """

    _containers: Dict[Union["str", uuid.UUID], CompositeEnvelopeContainer] = {}
    _instances: Dict[Union["str", uuid.UUID], List["CompositeEnvelope"]] = {}

    def __init__(
        self, *states: Union["CompositeEnvelope", Envelope, BaseState]
    ):
        self.uid = uuid.uuid4()
        # Check if there are composite envelopes in the argument list
        composite_envelopes: List[CompositeEnvelope] = [
            e for e in states if isinstance(e, CompositeEnvelope)
        ]
        envelopes = [e for e in states if _is_envelope(e)]
        state_objs: List[BaseState] = []
        for e in states:
            if _is_custom_state(e):
                state_objs.append(cast(BaseState, e))
            elif _is_envelope(e):
                state_objs.append(cast(BaseState, e.fock))
                state_objs.append(cast(BaseState, e.polarization))
        for e in envelopes:
            if (
                e.composite_envelope is not None
                and e.composite_envelope not in composite_envelopes
            ):
                composite_envelopes.append(e.composite_envelope)

        ce_container = None
        for ce in composite_envelopes:
            assert isinstance(
                ce, CompositeEnvelope
            ), "ce should be CompositeEnvelope type"
            state_objs.extend(ce.state_objs)
            if ce_container is None:
                ce_container = CompositeEnvelope._containers[ce.uid]
            else:
                ce_container.append_states(
                    CompositeEnvelope._containers[ce.uid]
                )
            ce.uid = self.uid
        if ce_container is None:
            ce_container = CompositeEnvelopeContainer(self.uid)
        for e in envelopes:
            if e not in ce_container.envelopes:
                assert e is not None, "Envelope e should not be None"
                if not _is_envelope(e):
                    raise TypeError("Expected an Envelope-like object")
                ce_container.envelopes.append(e)
        for s in state_objs:
            if s.uid not in [x.uid for x in ce_container.state_objs]:
                ce_container.state_objs.append(s)

        CompositeEnvelope._containers[self.uid] = ce_container
        if not CompositeEnvelope._instances.get(self.uid):
            CompositeEnvelope._instances[self.uid] = []
        CompositeEnvelope._instances[self.uid].append(self)
        self.update_composite_envelope_pointers()

    def __repr__(self) -> str:
        envelopes_repr = [e.uid for e in self.envelopes]
        states_repr = [s.uid for s in self.state_objs]
        return (
            f"CompositeEnvelope(uid={self.uid}, "
            f"envelopes={envelopes_repr},state_objects={states_repr})"
        )

    @property
    def envelopes(self) -> List["Envelope"]:
        return CompositeEnvelope._containers[self.uid].envelopes

    @property
    def state_objs(self) -> List["BaseState"]:
        return CompositeEnvelope._containers[self.uid].state_objs

    @property
    def product_states(self) -> List[ProductState]:
        return CompositeEnvelope._containers[self.uid].states

    @property
    def container(self) -> CompositeEnvelopeContainer:
        return CompositeEnvelope._containers[self.uid]

    @property
    def states(self) -> List[ProductState]:
        return CompositeEnvelope._containers[self.uid].states

    def update_composite_envelope_pointers(self) -> None:
        """
        Updates all the envelopes to point to this composite envelope
        """
        for envelope in self.envelopes:
            envelope.set_composite_envelope_id(self.uid)

    def expand(self, *states: "BaseState") -> None:
        product_states = [
            p for p in self.states if any(so in p.state_objs for so in states)
        ]
        for p in product_states:
            p.expand()

    def contract(self, *state_objs: "BaseState") -> None:
        pass

    # @timing_decorator
    def combine(self, *state_objs: "BaseState") -> None:
        """
        Combines given states into a product state.

        Parameters
        ----------
        state_objs: BaseState
           Accepts many state_objs
        """
        # stack = inspect.stack()
        # caller_frame = stack[1]
        # TODO
        # caller_function = caller_frame.function
        # caller_filename = caller_frame.filename
        # caller_line_number = caller_frame.lineno

        state_objs_set = set(state_objs)
        for ps in self.states:
            if state_objs_set.issubset(ps.state_objs):
                return  # Early exit on first match

        # Check if all states are included in composite envelope
        assert all(s in self.state_objs for s in state_objs)

        """
        Get all product states, which include any of the
        given states
        """
        existing_product_states = []
        for state in state_objs:
            for ps in self.product_states:
                if state in ps.state_objs:
                    existing_product_states.append(ps)
        # Removing duplicate product states
        # existing_product_states = list(set(existing_product_states))
        existing_product_states = list(dict.fromkeys(existing_product_states))

        """
        Ensure all states have the same expansion levels
        """
        minimum_expansion_level = ExpansionLevel.Vector
        for obj in state_objs:
            assert isinstance(obj.expansion_level, ExpansionLevel)
            if obj.expansion_level > minimum_expansion_level:
                minimum_expansion_level = ExpansionLevel.Matrix
                break

        # Expand the product spaces
        for product_state in existing_product_states:
            while product_state.expansion_level < minimum_expansion_level:
                product_state.expand()

        for obj in state_objs:
            if obj.index is None:
                assert isinstance(obj.expansion_level, ExpansionLevel)
                while obj.expansion_level < minimum_expansion_level:
                    obj.expand()
            elif isinstance(obj.index, int) and hasattr(obj, "envelope"):
                assert isinstance(obj.expansion_level, ExpansionLevel)
                while obj.expansion_level < minimum_expansion_level:
                    obj.expand()
        """
        Assemble all of the density matrices,
        and compile the indices in order
        """

        kron_arrays = []
        state_order = []

        # Collect arrays from new existing_product_states
        for product_state in existing_product_states:
            kron_arrays.append(product_state.state)
            state_order.extend(product_state.state_objs)
            product_state.state_objs = []
            product_state.state = jnp.array([[1]])

        # Collect arrays from new state objects
        for so in state_objs:
            if (
                hasattr(so, "envelope")
                and so.envelope is not None
                and so.index is not None
                and so.envelope.state is not None
                and not isinstance(so.index, tuple)
            ):
                kron_arrays.append(so.envelope.state)
                indices: List[Optional["BaseState"]] = [None, None]
                if isinstance(so.envelope.fock.index, int):
                    indices[so.envelope.fock.index] = so.envelope.fock
                if isinstance(so.envelope.polarization.index, int):
                    indices[so.envelope.polarization.index] = (
                        so.envelope.polarization
                    )
                state_order.extend(
                    [index for index in indices if index is not None]
                )
                so.envelope.state = None
            elif so.index is None:
                assert isinstance(so.state, jnp.ndarray)
                kron_arrays.append(so.state)
                state_order.append(so)
                so.state = None

        state_vector_or_matrix = kron_reduce(kron_arrays)
        """
        Create a new product state object and append it to the states
        """
        ps = ProductState(
            expansion_level=minimum_expansion_level,
            container=self._containers[self.uid],
            state=state_vector_or_matrix,
            state_objs=state_order,
        )

        CompositeEnvelope._containers[self.uid].states.append(ps)

        """
        Remove empty states
        """
        self.container.remove_empty_product_states()
        self.container.update_all_indices()

    def reorder(self, *ordered_states: "BaseState") -> None:
        """
        Changes the order of the states in the produce space
        If not all states are given, the given states will be
        put in the given order at the beginnig of the product
        states

        Parameters
        ----------
        *ordered_states: BaseState
            ordered list of states
        """
        # Check if given states are shared in a product space
        states_are_combined = False
        for ps in self.states:
            if all(s in ps.state_objs for s in ordered_states):
                states_are_combined

        # TODO
        # stack = inspect.stack()

        # caller_frame = stack[1]

        # caller_function = caller_frame.function
        # caller_filename = caller_frame.filename
        # caller_line_number = caller_frame.lineno
        if not states_are_combined:
            self.combine(*ordered_states)

        # Get the correct product state:
        ps = [
            p
            for p in self.states
            if all(so in p.state_objs for so in ordered_states)
        ][0]

        # Create order
        new_order = [s for s in ps.state_objs]
        for i, ordered_state in enumerate(ordered_states):
            if new_order.index(ordered_state) != i:
                tmp = new_order[i]
                old_idx = new_order.index(ordered_state)
                new_order[i] = ordered_state
                new_order[old_idx] = tmp
        ps.reorder(*new_order)

    def measure(
        self,
        *states: "BaseState",
        separate_measurement: bool = True,
        destructive: bool = True,
        key: jnp.ndarray | None = None,
        return_key: bool = False,
    ) -> Dict["BaseState", int] | Tuple[Dict["BaseState", int], jnp.ndarray | None]:
        """
        Measures subspace in this composite envelope. If the state is measured
        partially, then the state are moved to their respective spaces. If the
        measurement is destructive, then the state is destroyed post measurement.

        Parameter
        ---------
        states: Optional[BaseState]
            States that should be measured
        separate_measurement:bool
            If true given states will be measured separately. Has only affect on the
            envelope states. If state is part of the envelope and separate_measurement
            is True, then the given state will be measured separately and the other
            state in the envelope won't be measured
        destructive: bool
            If False, the measurement will not destroy the state after the measurement.
            The state will still be affected by the measurement (True by default)

        Returns
        -------
        Dict[BaseState,int]
            Dictionary of outcomes, where the state is key and its outcome measurement
            is the value (int)
        jnp.ndarray or None
            Next key after measurement (if provided and `return_key` is True).
        """
        C = Config()
        if key is None and C.use_jit:
            raise ValueError(
                "PRNG key is required when Config.use_jit is True for composite"
                "measurements"
            )
        if key is None:
            key = C.random_key
        outcomes: Dict["BaseState", int]
        outcomes = {}
        current_key = key
        next_key: jnp.ndarray | None = None

        # Compile the complete list of states
        state_list = list(states)
        if not separate_measurement:
            for s in state_list:
                env = getattr(s, "envelope", None)
                if env is not None and _is_envelope(env):
                    os: BaseState | None = None
                    if _is_fock(s):
                        os = cast(BaseState, env.polarization)
                    elif _is_polarization(s):
                        os = cast(BaseState, env.fock)
                    if os is not None and os not in state_list:
                        state_list.append(os)

        # If the state resides in the BaseState or Envelope measure there
        for s in state_list:
            if isinstance(s.index, int) and hasattr(s, "envelope"):
                env = s.envelope
                if env is None or not _is_envelope(env):
                    raise ValueError(
                        "State is missing envelope for measurement"
                    )
                use_key, current_key = borrow_key(current_key)
                if separate_measurement:
                    out, current_key = env.measure(
                        s,
                        separate_measurement=separate_measurement,
                        destructive=destructive,
                        key=use_key,
                        return_key=True,
                    )
                else:
                    os = env.fock if _is_polarization(s) else env.polarization
                    out, current_key = env.measure(
                        s,
                        os,
                        separate_measurement=separate_measurement,
                        destructive=destructive,
                        key=use_key,
                        return_key=True,
                    )
                for k, o in out.items():
                    outcomes[k] = o
            elif s.index is None:
                if not s.measured:
                    use_key, current_key = borrow_key(current_key)
                    out = s.measure(
                        separate_measurement=separate_measurement,
                        destructive=destructive,
                        key=use_key,
                    )
                    for k, o in out.items():
                        outcomes[k] = o

        # Measure in all of the product states
        product_states = [
            p
            for p in self.states
            if any(so in p.state_objs for so in state_list)
        ]

        for ps in product_states:
            ps_states = [so for so in state_list if so in ps.state_objs]
            use_key, current_key = borrow_key(current_key)
            out, ps_next_key = ps.measure(
                *ps_states,
                separate_measurement=separate_measurement,
                destructive=destructive,
                key=use_key,
                return_key=True,
            )
            next_key = ps_next_key
            for key, item in out.items():
                outcomes[key] = item

        if destructive:
            for s in state_list:
                env = getattr(s, "envelope", None)
                if env is not None and _is_envelope(env):
                    env._set_measured()

        self._containers[self.uid].update_all_indices()
        self._containers[self.uid].remove_empty_product_states()
        if return_key:
            return outcomes, next_key
        return outcomes

    def measure_expectation(
        self,
        *states: "BaseState",
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Differentiable expectation of measuring the given subsystems across the
        composite envelope. Returns (probabilities, expected post-measurement
        density matrix).
        """
        # Get product states containing targets
        if len(states) == 0:
            assert len(self.product_states) > 0
            ps = self.product_states[0]
            targets: Tuple[BaseState, ...] = tuple(ps.state_objs)
        else:
            product_states = [
                p
                for p in self.states
                if any(so in p.state_objs for so in states)
            ]
            if len(product_states) > 1:
                all_states: List[BaseState] = list(states)
                for p in product_states:
                    all_states.extend([s for s in p.state_objs])
                self.combine(*all_states)
                product_states = [
                    p
                    for p in self.states
                    if any(so in p.state_objs for so in states)
                ]
            if len(product_states) == 0:
                self.combine(*states)
                product_states = [
                    p
                    for p in self.states
                    if any(so in p.state_objs for so in states)
                ]
            assert len(product_states) > 0
            ps = product_states[0]
            targets = tuple(states)

        return ps.measure_expectation(*targets)

    def measure_POVM(
        self,
        operators: List[jnp.ndarray],
        *states: "BaseState",
        destructive: bool = True,
        key: jnp.ndarray | None = None,
    ) -> Tuple[int, Dict["BaseState", int]]:
        """
        Perform a POVM measurement.

        Parameters
        ----------
        operators: List[jnp.ndarray]
            List of POVM operators
        *states: List[Union[Fock, Polarization]]
            List of states, on which the POVM measurement should be executed
            The order of the states must reflect the order of tensoring of
            individual hilbert space in the individual operator
        destructive: bool
            If desctructive is set to True, then the states will be
            destroyed post measurement

        Returns
        -------
        int: Outcome result of index
        """

        # Check if the operator dimensions match
        dim = jnp.prod(jnp.array([s.dimensions for s in states]))
        for op in operators:
            if op.shape != (dim, dim):
                raise ValueError(
                    "At least on Kraus operator has incorrect dimensions: "
                    f"{op.shape}, expected({dim},{dim})"
                )

        current_key = key
        # Get product states
        product_states = [
            p for p in self.states if any(so in p.state_objs for so in states)
        ]
        ps = None
        if len(product_states) > 1:
            all_states = [s for s in states]
            for p in product_states:
                all_states.extend([s for s in p.state_objs])
            self.combine(*all_states)
            ps = [
                p
                for p in self.states
                if any(so in p.state_objs for so in states)
            ][0]
        elif len(product_states) == 0:
            self.combine(*states)
            ps = [
                p
                for p in self.states
                if any(so in p.state_objs for so in states)
            ][0]
        else:
            ps = product_states[0]
        # Make sure the order of the states in tensoring is correct
        self.reorder(*states)

        outcome = ps.measure_POVM(
            operators, *states, destructive=destructive, key=current_key
        )
        return outcome

    def apply_kraus(
        self,
        operators: List[jnp.ndarray],
        *states: "BaseState",
        identity_check: bool = True,
    ) -> None:
        """
        Apply kraus operator to the given states
        The product state is automatically expanded to the density matrix
        representation. The order of tensoring in the operators should be the same
        as the order of the given states. The order of the states in the product
        state is changed to reflect the order of the given states.

        Parameters
        ----------
        operators: List[jnp.ndarray]
            List of all Kraus operators
        *states: BaseState
            List of the states, that the channel should be applied to
        identity_check: bool
            True by default, if true the method checks if kraus condition holds
        """

        # Check the uniqueness of the states
        if len(states) != len(list(set(states))):
            raise ValueError("State list should contain unique elements")

        # Check if dimensions match
        dim = jnp.prod(jnp.array([s.dimensions for s in states]))
        for op in operators:
            if op.shape != (dim, dim):
                raise ValueError(
                    "At least on Kraus operator has incorrect dimensions: "
                    f"{op.shape}, expected({dim},{dim})"
                )

        # Check the identity sum
        if identity_check:
            if not kraus_identity_check(operators):
                raise ValueError("Kraus operators do not sum to the identity")
        # Get product states
        product_states = [
            p for p in self.states if any(so in p.state_objs for so in states)
        ]
        ps = None
        if len(product_states) > 1:
            all_states = [s for s in states]
            for p in product_states:
                all_states.extend([s for s in p.state_objs])
            self.combine(*all_states)
            ps = [
                p
                for p in self.states
                if any(so in p.state_objs for so in states)
            ][0]
        elif len(product_states) == 0:
            # If only one state has to have kraus operators applied and the
            # state is not combined apply the kraus there
            if len(states) == 1:
                states[0].apply_kraus(operators)
                return
            elif len(states) == 2:
                if hasattr(states[0], "envelope") and hasattr(
                    states[1], "envelope"
                ):
                    if states[0].envelope == states[1].envelope:
                        env = states[0].envelope
                        if env is not None and _is_envelope(env):
                            env.apply_kraus(operators, *states)
                            return
            self.combine(*states)
            ps = [
                p
                for p in self.states
                if any(so in p.state_objs for so in states)
            ][0]
        else:
            ps = product_states[0]
        # Make sure the order of the states in tensoring is correct
        self.reorder(*states)

        ps.apply_kraus(operators, *states)

    def trace_out(self, *states: Union["BaseState"]) -> jnp.ndarray:
        """
        Traces out the rest of the states from the product state and returns
        the resultint matrix or vector. If given states are in sparate product
        spaces it merges the product spaces.

        Parameters
        ----------
        *states: Union[Fock, Polarization]

        Returns
        -------
        jnp.ndarray
            Traced out system including only the requested states in tesored
            in the order in which the states are given
        """
        product_states = [
            p for p in self.states if any(so in p.state_objs for so in states)
        ]
        assert len(product_states) > 0, "No product state found"
        ps: ProductState
        if len(product_states) > 1:
            all_states = [s for s in states]
            for p in product_states:
                all_states.extend([s for s in p.state_objs])
            self.combine(*all_states)
            product_states = [
                p
                for p in self.states
                if any(so in p.state_objs for so in states)
            ]
            assert (
                len(product_states) > 0
            ), "Only one product state should exist at this point"
        ps = product_states[0]

        if len(states) > 1:
            self.reorder(*states)

        to = ps.trace_out(*states)

        return to

    def resize_fock(self, new_dimensions: int, fock: BaseState) -> bool:
        """
        Resizes the space to the new dimensions.
        If the dimensions are more, than the current dimensions, then
        it gets padded. If the dimensions are less then the current
        dimensions, then it checks if it can shrink the space.

        Parameters
        ----------
        new_dimensions: bool
            New dimensions to be set
        fock: Fock
            The fock state, for which the dimensions
            must be changed

        Returns
        -------
        bool
            True if the resizing was succesfull
        """
        # Check if fock is Fock type
        if not _is_fock(fock):
            raise ValueError("Only Fock spaces can be resized")

        # Check if fock is in this composite envelope
        if fock not in self.state_objs:
            raise ValueError(
                "Tried to resizing fock, which is not a part of this envelope"
            )

        if not isinstance(fock.index, tuple):
            return fock.resize(new_dimensions)

        ps = [ps for ps in self.product_states if fock in ps.state_objs]
        if len(ps) != 1:
            raise ValueError("Something went wrong")  # pragma : no cover

        new_ps = ps[0]
        self.reorder(fock)
        return new_ps.resize_fock(new_dimensions, fock)

    def apply_operation(self, operator: Operation, *states: BaseState) -> None:
        """
        Applies the operation to the correct product space. If operator
        has type CompositeOperator, then the product states are joined if
        the states are not yet in the same product state

        Parameters
        ----------
        operator: Operator
            Operator which should be appied to the state(s)
        states: BaseState
            States onto which the operator should be applied
        """
        if len(states) == 1:
            if not isinstance(states[0].index, tuple):
                assert hasattr(states[0], "apply_operation")
                states[0].apply_operation(operator)
                return
        product_states = [
            p for p in self.states if any(so in p.state_objs for so in states)
        ]
        ps = None
        if len(product_states) > 1 or (
            len(product_states) == 1
            and not all(
                state in product_states[0].state_objs for state in states
            )
        ):
            all_states = [s for s in states]
            for p in product_states:
                all_states.extend([s for s in p.state_objs])
            self.combine(*all_states)
            ps = [
                p
                for p in self.states
                if any(so in p.state_objs for so in states)
            ][0]
        elif len(product_states) == 1:
            ps = product_states[0]
        if len(product_states) == 0:
            all_states = [s for s in states]
            self.combine(*all_states)
            ps = [
                p
                for p in self.states
                if any(so in p.state_objs for so in states)
            ][0]

        assert isinstance(ps, ProductState)
        ps.apply_operation(operator, *states)

    def prepare_meta(
        self, product_state_index: int = 0, *target_states: BaseState
    ) -> shape_planning.DimsMeta:
        r"""
        Prepare static dimension metadata for a product state in this composite
        envelope.

        If no target states are provided, all states in the selected product state are
        targets.
        """
        return shape_planning.composite_meta(
            self, product_state_index, *target_states
        )

    def compiled_kernels(
        self, product_state_index: int = 0, *target_states: BaseState
    ):
        """
        Return meta-bound, jitted kernels for a product state in this composite
        envelope.

        If no target states are provided, all states in the selected product state
        are targets.
        """
        return shape_planning.compiled_kernels(
            self.prepare_meta(product_state_index, *target_states)
        )
