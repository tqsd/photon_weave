import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
import numpy as np

# from photon_weave.extra.einsum_constructor import EinsumStringConstructor as ESC
import photon_weave.extra.einsum_constructor as ESC
from photon_weave._math.ops import (
    kraus_identity_check,
    num_quanta_matrix,
    num_quanta_vector,
)
from photon_weave.operation import (
    CustomStateOperationType,
    FockOperationType,
    Operation,
    PolarizationOperationType,
)
from photon_weave.photon_weave import Config
from photon_weave.state.expansion_levels import ExpansionLevel

# For static type checks
if TYPE_CHECKING:
    from photon_weave.state.base_state import BaseState  # pragma: no cover
    from photon_weave.state.custom_state import CustomState  # pragma: no cover
    from photon_weave.state.envelope import Envelope  # pragma: no cover
    from photon_weave.state.fock import Fock  # pragma: no cover


@dataclass(slots=True)
class ProductState:
    """
    Stores Product state and references to its constituents
    """

    expansion_level: ExpansionLevel
    container: "CompositeEnvelopeContainer"
    uid: uuid.UUID = field(default_factory=uuid.uuid4)
    state: jnp.ndarray = field(default_factory=lambda: jnp.array([[1]]))
    state_objs: List["BaseState"] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.uid)

    def expand(self) -> None:
        """
        Expands the state from vector to matrix
        """
        if self.expansion_level < ExpansionLevel.Matrix:
            self.state = jnp.outer(self.state.flatten(), jnp.conj(self.state.flatten()))
            self.expansion_level = ExpansionLevel.Matrix
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
            # If the state is mixed, return
            if jnp.abs(jnp.trace(jnp.matmul(self.state, self.state)) - 1) >= tol:
                return

            eigenvalues, eigenvectors = jnp.linalg.eigh(self.state)
            pure_state_index = jnp.argmax(jnp.abs(eigenvalues - 1.0) < tol)
            assert pure_state_index is not None, "Pure state indx should not be None"
            self.state = eigenvectors[:, pure_state_index].reshape(-1, 1)
            for state in self.state_objs:
                state.expansion_level = ExpansionLevel.Vector
            self.expansion_level = ExpansionLevel.Vector

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
            state = jnp.einsum(einsum, state)

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
            state = jnp.einsum(einsum, state)

            # Reshape and reorder self.state_objs to reflect the new order
            new_dims = jnp.prod(jnp.array([s.dimensions for s in ordered_states]))
            self.state = state.reshape((new_dims, new_dims))
            self.state_objs = list(ordered_states)

        # Update indices in all of the states in this product space
        self.container.update_all_indices()

    def measure(
        self,
        *states: "BaseState",
        separate_measurement: bool = False,
        destructive: bool = True,
    ) -> Dict["BaseState", int]:
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
        """
        from photon_weave.state.polarization import Polarization, PolarizationLabel

        assert all(
            so in self.state_objs for so in states
        ), "All state objects need to be in product state"
        outcomes = {}
        C = Config()

        remaining_states = [s for s in self.state_objs]

        if self.expansion_level == ExpansionLevel.Vector:
            # Get the state and reshape it into tensor
            shape = [so.dimensions for so in self.state_objs]
            shape.append(1)
            ps = self.state.reshape(shape)
            for idx, state in enumerate(states):
                # Constructing the einsum str
                einsum = ESC.measure_vector(remaining_states, [state])

                # Project the state with einsum string
                projected_state = jnp.einsum(einsum, ps)

                # Outcome Probabilities
                probabilities = jnp.abs(projected_state.flatten()) ** 2
                probabilities /= jnp.sum(probabilities)

                # Decide on output
                key = C.random_key
                outcomes[state] = int(
                    jax.random.choice(
                        key, a=jnp.array(list(range(state.dimensions))), p=probabilities
                    )
                )

                # Construct the post measurement state
                # Even if the state is measured non_destructively
                # the state is removed and placed back into original
                # Base state as a label form
                indices: List[Union[slice, int]] = [slice(None)] * len(ps.shape)
                indices[remaining_states.index(state)] = outcomes[state]
                ps = ps[tuple(indices)]

                # Remove the measured state form the product state
                remaining_states.remove(state)

                # Handle post measurement processes
                if destructive:
                    state._set_measured()
                else:
                    if isinstance(state, Polarization):
                        if outcomes[state] == 0:
                            state.state = PolarizationLabel.H
                        else:
                            state.state = PolarizationLabel.V
                    else:
                        state.state = outcomes[state]
                    state.index = None
                    state.expansion_level = ExpansionLevel.Label
                self.state_objs.remove(state)
            if len(self.state_objs) > 0:
                # Handle reshaping and storing the post measurement product state
                self.state = ps.reshape(-1, 1)
                self.state /= jnp.linalg.norm(self.state)
            else:
                # Product state will be deleted
                self.state = jnp.array([[1]])
        elif self.expansion_level == ExpansionLevel.Matrix:
            shape = [so.dimensions for so in self.state_objs] * 2
            ps = self.state.reshape(shape)
            for idx, state in enumerate(states):
                # Generate einsum string
                einsum = ESC.measure_matrix(remaining_states, [state])

                # Project the state with einsum
                projected_state = jnp.einsum(einsum, ps)

                # Outcome Probabilities
                probabilities = jnp.abs(jnp.diag(projected_state))
                probabilities /= sum(probabilities)

                # Decide on outcome
                key = C.random_key
                outcomes[state] = int(
                    jax.random.choice(
                        key, a=jnp.array(list(range(state.dimensions))), p=probabilities
                    )
                )

                # Construct post measurement state
                # Even if state is measured in non_destructive manner
                # It is removed from the product state and placed back
                # into the BaseState
                state_index = remaining_states.index(state)
                row_idx = state_index
                col_idx = state_index + len(remaining_states)
                indices = [slice(None)] * len(ps.shape)
                indices[row_idx] = outcomes[state]
                indices[col_idx] = outcomes[state]
                ps = ps[tuple(indices)]

                # Remove the mesaured state from the remaining states
                remaining_states.remove(state)
                if destructive:
                    state._set_measured()
                else:
                    if isinstance(state, Polarization):
                        if outcomes[state] == 0:
                            state.state = PolarizationLabel.H
                        else:
                            state.state = PolarizationLabel.V
                    else:
                        state.state = outcomes[state]
                    state.index = None
                    state.expansion_level = ExpansionLevel.Label

                # Remove the mesaured state from the product state
                self.state_objs.remove(state)

            # Reconstruct the post measurement product state
            if len(self.state_objs) > 0:
                ps = ps.flatten()
                num_elements = ps.size
                sqrt = int(jnp.ceil(jnp.sqrt(num_elements)))
                self.state = ps.reshape((sqrt, sqrt))
                self.state /= jnp.linalg.norm(self.state)
            else:
                self.state = jnp.array([[1]])

        # Attempt to contract to Vector if set so
        if C.contractions:
            self.contract()
        return outcomes

    def measure_POVM(
        self,
        operators: List[Union[np.ndarray, jnp.ndarray]],
        *states: "BaseState",
        destructive: bool = True,
    ) -> Tuple[int, Dict["BaseState", int]]:
        """
        Perform a POVM measurement.

        Parameters
        ----------
        operators: List[Union[np.ndarray, jnp.ndarray]]
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

        # Transform the operators to the tensors
        op_shape = [s.dimensions for s in states] * 2

        # Reshape operators
        operators = [op.reshape(op_shape) for op in operators]

        # Get and reshape the state
        ps = self.state.reshape([s.dimensions for s in self.state_objs] * 2)

        # Generate the operators for application of operators
        einsum = ESC.apply_operator_matrix(self.state_objs, list(states))

        # Get the probabilities
        prob_list: List[float] = []

        # Get the dimensions of the measured states
        to_dims = jnp.prod(jnp.array([so.dimensions for so in states]))

        for op in operators:
            # Apply each operator
            prob_state = jnp.einsum(einsum, op, ps, jnp.conj(op))

            # Trace out the state to get the probabilities
            einsum_to = ESC.trace_out_matrix(self.state_objs, list(states))
            prob_state_to = jnp.einsum(einsum_to, prob_state).reshape(
                (to_dims, to_dims)
            )

            # Compute the outcome probability
            prob_list.append(float(jnp.trace(prob_state_to)))

        # Normalize the probabilities
        probabilities = jnp.array(prob_list)
        probabilities /= jnp.sum(probabilities)

        # Decide on the outcomes
        C = Config()
        key = C.random_key
        outcome = int(
            jax.random.choice(
                key,
                a=jnp.array(list(range(len(operators)))),
                p=jnp.array(probabilities),
            )
        )

        # Construct Post Measurement state
        new_dims = jnp.prod(jnp.array([s.dimensions for s in self.state_objs]))
        ps = jnp.einsum(
            einsum, operators[outcome], ps, jnp.conj(operators[outcome])
        ).reshape((new_dims, new_dims))
        self.state = ps / jnp.trace(ps)
        other_outcomes = {}
        if destructive:
            # Get correct Composite Envelope
            if isinstance(
                CompositeEnvelope._instances[self.container.composite_uid], list
            ):
                other_outcomes = CompositeEnvelope._instances[
                    self.container.composite_uid
                ][0].measure(*states)
                for s in states:
                    del other_outcomes[s]
        if C.contractions:
            self.contract()
        return (outcome, other_outcomes)

    def apply_kraus(
        self, operators: List[Union[np.ndarray, jnp.ndarray]], *states: "BaseState"
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
        if self.expansion_level == ExpansionLevel.Vector:
            # Get the state and reshape it
            shape = [s.dimensions for s in self.state_objs]
            shape.append(1)
            ps = self.state.reshape(shape)

            # Reshape the operators
            op_shape = [s.dimensions for s in states] * 2
            operators = [op.reshape(op_shape) for op in operators]

            # Generate einsum
            einsum = ESC.apply_operator_vector(self.state_objs, list(states))

            # Apply all operators
            resulting_state = jnp.zeros_like(ps)

            for op in operators:
                resulting_state += jnp.einsum(einsum, op, ps)

            # Reshape the resulting state back into vector and store it
            self.state = resulting_state.reshape(-1, 1)

        elif self.expansion_level == ExpansionLevel.Matrix:
            # Get the state and reshape it
            ps = self.state.reshape([s.dimensions for s in self.state_objs] * 2)

            # Reshape the operators
            op_shape = [s.dimensions for s in states] * 2
            operators = [op.reshape(op_shape) for op in operators]

            # Genetate einsum
            einsum = ESC.apply_operator_matrix(self.state_objs, list(states))

            # Generate a new state to sum into
            resulting_state = jnp.zeros_like(ps)

            # Apply all operators
            for op in operators:
                resulting_state += jnp.einsum(einsum, op, ps, jnp.conj(op))

            # Compute matrix dimensions
            dims = jnp.prod(jnp.array([s.dimensions for s in self.state_objs]))

            # Change the state back to the matrix form
            self.state = resulting_state.reshape(dims, dims)
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
        if self.expansion_level == ExpansionLevel.Vector:
            # Reshape the vector into tensor
            shape = [s.dimensions for s in self.state_objs] + [1]
            ps = self.state.reshape(shape)

            # Compute einsum string
            einsum = ESC.trace_out_vector(self.state_objs, list(states))

            # Perform the tracing
            traced_out_state = jnp.einsum(einsum, ps)

            # Reshape and return
            return traced_out_state.reshape((-1, 1))
        elif self.expansion_level == ExpansionLevel.Matrix:
            # Reshape the matrix into tensor
            ps = self.state.reshape([s.dimensions for s in self.state_objs] * 2)

            # Generate einsum string
            einsum = ESC.trace_out_matrix(self.state_objs, list(states))

            # Perform the tracing
            traced_out_state = jnp.einsum(einsum, ps)

            # Compute the new dimensions
            new_dims = jnp.prod(jnp.array([s.dimensions for s in states]))

            # Reshape and Return
            return traced_out_state.reshape((new_dims, new_dims))
        else:
            raise ValueError("Something went wrong")  # pragma: no cover

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

    def resize_fock(self, new_dimensions: int, fock: "Fock") -> bool:
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
                ps = jnp.pad(ps, pad_config, mode="constant", constant_values=0)
                fock.dimensions = new_dimensions
                dims = jnp.prod(jnp.array([s.dimensions for s in self.state_objs]))
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
                dims = jnp.prod(jnp.array([s.dimensions for s in self.state_objs]))
                self.state = ps.reshape((dims, 1))
                return True

        if self.expansion_level == ExpansionLevel.Matrix:
            shape = [so.dimensions for so in self.state_objs]
            transpose_pattern = [
                item
                for i in range(len(self.state_objs))
                for item in [i, i + len(self.state_objs)]
            ]
            ps = self.state.reshape([*shape, *shape]).transpose(transpose_pattern)
            if new_dimensions > fock.dimensions:
                assert isinstance(fock.index, tuple)
                padding = new_dimensions - fock.dimensions
                pad_config = [(0, 0) for _ in range(ps.ndim)]
                pad_config[fock.index[1] * len(self.state_objs)] = (0, padding)
                pad_config[fock.index[1] * len(self.state_objs) + 1] = (0, padding)
                ps = jnp.pad(ps, pad_config, mode="constant", constant_values=0)
                fock.dimensions = new_dimensions
                dims = jnp.prod(jnp.array([s.dimensions for s in self.state_objs]))
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
                slices[fock.index[1] * len(self.state_objs)] = slice(0, new_dimensions)
                slices[fock.index[1] * len(self.state_objs) + 1] = slice(
                    0, new_dimensions
                )
                ps = ps[tuple(slices)]
                fock.dimensions = new_dimensions
                ps = ps.transpose(transpose_pattern)
                dims = jnp.prod(jnp.array([s.dimensions for s in self.state_objs]))
                self.state = jnp.array(ps.reshape((dims, dims)))
                return True
        return False

    def apply_operation(self, operation: Operation, *states: "BaseState") -> None:
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
        from photon_weave.operation import CompositeOperationType
        from photon_weave.state.custom_state import CustomState
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        if isinstance(operation._operation_type, FockOperationType):
            assert isinstance(states[0], Fock)
            assert len(states) == 1
            to = states[0].trace_out()
            assert isinstance(to, jnp.ndarray)
            operation.compute_dimensions(states[0]._num_quanta, to)
            states[0].resize(operation.dimensions[0])
        elif isinstance(operation._operation_type, PolarizationOperationType):
            assert isinstance(states[0], Polarization)
            assert len(states) == 1
            # Parameters doesn't have any effect
            operation.compute_dimensions(0, jnp.array([0]))
        elif isinstance(operation._operation_type, CustomStateOperationType):
            assert isinstance(states[0], CustomState)
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
            operation.compute_dimensions(
                [s._num_quanta if isinstance(s, Fock) else 0 for s in states],
                [s.trace_out() for s in states],  # type: ignore
            )
            for i, s in enumerate(states):
                if isinstance(s, Fock):
                    s.resize(operation._dimensions[i])

        shape = [so.dimensions for so in self.state_objs]
        if self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            dims = jnp.prod(jnp.array([so.dimensions for so in self.state_objs]))
            assert self.state.shape == (dims, 1)

            # Get the Reshaped state
            shape.append(1)
            ps = self.state.reshape(shape)

            # Get the operator and Reshape it
            operator = operation.operator.reshape([s.dimensions for s in states] * 2)

            # Generate the Einstein sum string
            einsum = ESC.apply_operator_vector(self.state_objs, list(states))

            # Constructing einsum string
            ps = jnp.einsum(einsum, operator, ps)

            if not jnp.any(jnp.abs(ps) > 0):
                raise ValueError(
                    "The state is entirely composed of zeros, "
                    "is |0⟩ attempted to be annihilated?"
                )
            if operation.renormalize:
                ps = ps / jnp.linalg.norm(ps)
            self.state = ps.reshape((-1, 1))
        elif self.expansion_level == ExpansionLevel.Matrix:
            assert isinstance(self.state, jnp.ndarray)
            dims = jnp.prod(jnp.array([so.dimensions for so in self.state_objs]))
            assert self.state.shape == (dims, dims)

            # Get the State and Reshape it
            ps = self.state.reshape([*shape, *shape])

            # Get the operator and Reshape it
            operator = operation.operator.reshape([s.dimensions for s in states] * 2)

            # Generate Einstein sum string
            einsum = ESC.apply_operator_matrix(self.state_objs, list(states))

            # Apply the Einstein Summation
            ps = jnp.einsum(einsum, operator, ps, jnp.conj(operator))

            if not jnp.any(jnp.abs(ps) > 0):
                raise ValueError(
                    "The state is entirely composed of zeros,"
                    "is |0⟩ attempted to be annihilated?"
                )
            if operation.renormalize:
                ps = ps / jnp.linalg.norm(ps)

            # Reshape back into 2d Matrix
            dims = jnp.prod(jnp.array([so.dimensions for so in self.state_objs]))
            self.state = ps.reshape((dims, dims))
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
        Appnds the states of two composite envelope containers
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

    def __init__(self, *states: Union["CompositeEnvelope", "Envelope", "CustomState"]):
        from photon_weave.state.base_state import BaseState
        from photon_weave.state.custom_state import CustomState
        from photon_weave.state.envelope import Envelope
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        self.uid = uuid.uuid4()
        # Check if there are composite envelopes in the argument list
        composite_envelopes: List[CompositeEnvelope] = [
            e for e in states if isinstance(e, CompositeEnvelope)
        ]
        envelopes: List[Envelope] = [e for e in states if isinstance(e, Envelope)]
        state_objs: List[Union[CustomState, Fock, Polarization, BaseState]] = []
        for e in states:
            if isinstance(e, CustomState):
                state_objs.append(e)
            elif isinstance(e, Envelope):
                state_objs.append(e.fock)
                state_objs.append(e.polarization)
        for e in envelopes:
            if (
                e.composite_envelope is not None
                and e.composite_envelope not in composite_envelopes
            ):
                assert isinstance(
                    e.composite_envelope, CompositeEnvelope
                ), "e.composite_envelope should be CompositeEnvelope type"
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
                ce_container.append_states(CompositeEnvelope._containers[ce.uid])
            ce.uid = self.uid
        if ce_container is None:
            ce_container = CompositeEnvelopeContainer(self.uid)
        for e in envelopes:
            if e not in ce_container.envelopes:
                assert e is not None, "Envelope e should not be None"
                assert isinstance(e, Envelope), "e should be of type Envelope"
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
        return (
            f"CompositeEnvelope(uid={self.uid}, "
            f"envelopes={[e.uid for e in self.envelopes]}, "
            f"state_objects={[s.uid for s in self.state_objs]})"
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

    def combine(self, *state_objs: "BaseState") -> None:
        """
        Combines given states into a product state.

        Parameters
        ----------
        state_objs: BaseState
           Accepts many state_objs
        """
        from photon_weave.state.base_state import BaseState
        from photon_weave.state.envelope import Envelope

        # Check for the types
        for so in state_objs:
            assert isinstance(so, BaseState), f"got {type(so)}, expected BaseState"

        # Check if the states are already in the same product space
        for ps in self.states:
            if all(s in ps.state_objs for s in state_objs):
                return

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
                    assert isinstance(obj.envelope, Envelope)
                    obj.envelope.expand()

        """
        Assemble all of the density matrices,
        and compile the indices in order
        """
        state_vector_or_matrix = jnp.array([[1]])
        state_order = []
        target_state_objs = [so for so in state_objs]
        for product_state in existing_product_states:
            state_vector_or_matrix = jnp.kron(
                state_vector_or_matrix, product_state.state
            )
            state_order.extend(product_state.state_objs)
            product_state.state_objs = []
        for so in target_state_objs:
            if (
                hasattr(so, "envelope")
                and so.envelope is not None
                and so.index is not None
                and not isinstance(so.index, tuple)
            ):
                assert isinstance(so.envelope.state, jnp.ndarray)
                if minimum_expansion_level is ExpansionLevel.Vector:
                    assert so.envelope.state.shape == (so.envelope.dimensions, 1)
                    state_vector_or_matrix = jnp.kron(
                        state_vector_or_matrix, so.envelope.state
                    )
                    so.envelope.state = None
                else:
                    assert so.envelope.state.shape == (
                        so.envelope.dimensions,
                        so.envelope.dimensions,
                    )
                    state_vector_or_matrix = jnp.kron(
                        state_vector_or_matrix, so.envelope.state
                    )
                    so.envelope.state = None
                indices: List[Optional[BaseState]] = [None, None]
                assert isinstance(so.envelope.fock.index, int)
                assert isinstance(so.envelope.polarization.index, int)
                indices[so.envelope.fock.index] = so.envelope.fock
                indices[so.envelope.polarization.index] = so.envelope.polarization
                assert all(
                    x is not None for x in indices
                ), "Indices must not be None at this point"
                state_order.extend(cast(List["BaseState"], indices))
            if so.index is None:
                assert isinstance(so.state, jnp.ndarray)
                if minimum_expansion_level is ExpansionLevel.Vector:
                    assert so.state.shape == (so.dimensions, 1)
                    state_vector_or_matrix = jnp.kron(state_vector_or_matrix, so.state)
                else:
                    assert so.state.shape == (so.dimensions, so.dimensions)
                    state_vector_or_matrix = jnp.kron(state_vector_or_matrix, so.state)
                so.state = None
                state_order.append(so)

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
        if not states_are_combined:
            self.combine(*ordered_states)

        # Get the correct product state:
        ps = [
            p for p in self.states if all(so in p.state_objs for so in ordered_states)
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
        separate_measurement: bool = False,
        destructive: bool = True,
    ) -> Dict["BaseState", int]:
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
        """
        from photon_weave.state.envelope import Envelope
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        outcomes: Dict["BaseState", int]
        outcomes = {}

        # Compile the complete list of states
        state_list = list(states)
        if not separate_measurement:
            for s in state_list:
                if hasattr(s, "envelope"):
                    assert isinstance(s.envelope, Envelope)
                    os: Union[Fock, Polarization]
                    if isinstance(s, Fock):
                        os = s.envelope.polarization
                    elif isinstance(s, Polarization):
                        os = s.envelope.fock
                    if os not in state_list:
                        state_list.append(os)

        # If the state resides in the BaseState or Envelope measure there
        for s in state_list:
            if isinstance(s.index, int) and hasattr(s, "envelope"):
                assert isinstance(s.envelope, Envelope)
                if separate_measurement:
                    out = s.envelope.measure(
                        s,
                        separate_measurement=separate_measurement,
                        destructive=destructive,
                    )
                else:
                    os = (
                        s.envelope.fock
                        if isinstance(s, Polarization)
                        else s.envelope.polarization
                    )
                    out = s.envelope.measure(
                        s,
                        os,
                        separate_measurement=separate_measurement,
                        destructive=destructive,
                    )
                for k, o in out.items():
                    outcomes[k] = o
            elif s.index is None:
                if not s.measured:
                    out = s.measure(
                        separate_measurement=separate_measurement,
                        destructive=destructive,
                    )
                    for k, o in out.items():
                        outcomes[k] = o

        # Measure in all of the product states
        product_states = [
            p for p in self.states if any(so in p.state_objs for so in state_list)
        ]

        for ps in product_states:
            ps_states = [so for so in state_list if so in ps.state_objs]
            out = ps.measure(
                *ps_states,
                separate_measurement=separate_measurement,
                destructive=destructive,
            )
            for key, item in out.items():
                outcomes[key] = item

        for s in state_list:
            if hasattr(s, "envelope") and destructive:
                assert isinstance(s.envelope, Envelope)
                s.envelope._set_measured()

        self._containers[self.uid].update_all_indices()
        self._containers[self.uid].remove_empty_product_states()
        return outcomes

    def measure_POVM(
        self,
        operators: List[Union[np.ndarray, jnp.ndarray]],
        *states: "BaseState",
        destructive: bool = True,
    ) -> Tuple[int, Dict["BaseState", int]]:
        """
        Perform a POVM measurement.

        Parameters
        ----------
        operators: List[Union[np.ndarray, jnp.ndarray]]
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
            ps = [p for p in self.states if any(so in p.state_objs for so in states)][0]
        elif len(product_states) == 0:
            self.combine(*states)
            ps = [p for p in self.states if any(so in p.state_objs for so in states)][0]
        else:
            ps = product_states[0]
        # Make sure the order of the states in tensoring is correct
        self.reorder(*states)

        outcome = ps.measure_POVM(operators, *states, destructive=destructive)
        return outcome

    def apply_kraus(
        self,
        operators: List[Union[np.ndarray, jnp.ndarray]],
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
        operators: List[Union[np.ndarray, jnp.ndarray]]
            List of all Kraus operators
        *states: BaseState
            List of the states, that the channel should be applied to
        identity_check: bool
            True by default, if true the method checks if kraus condition holds
        """

        from photon_weave.state.envelope import Envelope
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

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
            ps = [p for p in self.states if any(so in p.state_objs for so in states)][0]
        elif len(product_states) == 0:
            # If only one state has to have kraus operators applied and the
            # state is not combined apply the kraus there
            if len(states) == 1:
                states[0].apply_kraus(operators)
                return
            elif len(states) == 2:
                if hasattr(states[0], "envelope") and hasattr(states[1], "envelope"):
                    if states[0].envelope == states[1].envelope:
                        assert isinstance(states[0].envelope, Envelope)
                        for s in states:
                            assert isinstance(s, (Fock, Polarization))
                        states[0].envelope.apply_kraus(operators, *states)
                        return
            self.combine(*states)
            ps = [p for p in self.states if any(so in p.state_objs for so in states)][0]
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
                p for p in self.states if any(so in p.state_objs for so in states)
            ]
            assert (
                len(product_states) > 0
            ), "Only one product state should exist at this point"
        ps = product_states[0]

        self.reorder(*states)

        return ps.trace_out(*states)

    def resize_fock(self, new_dimensions: int, fock: "Fock") -> bool:
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
        from photon_weave.state.fock import Fock

        # Check if fock is Fock type
        if not isinstance(fock, Fock):
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

    def apply_operation(
        self, operator: Operation, *states: Union["BaseState", "CustomState"]
    ) -> None:
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
            and not all(state in product_states[0].state_objs for state in states)
        ):
            all_states = [s for s in states]
            for p in product_states:
                all_states.extend([s for s in p.state_objs])
            self.combine(*all_states)
            ps = [p for p in self.states if any(so in p.state_objs for so in states)][0]
        elif len(product_states) == 1:
            ps = product_states[0]
        if len(product_states) == 0:
            all_states = [s for s in states]
            self.combine(*all_states)
            ps = [p for p in self.states if any(so in p.state_objs for so in states)][0]

        assert isinstance(ps, ProductState)
        ps.apply_operation(operator, *states)
