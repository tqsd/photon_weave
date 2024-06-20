import numpy as np

from .expansion_levels import ExpansionLevel


class FockOrPolarizationExpectedException(Exception):
    pass


class StateNotInThisCompositeEnvelopeException(Exception):
    pass


def redirect_if_consumed(method):
    def wrapper(self, *args, **kwargs):
        # Check if the object has been consumed by another CompositeEnvelope
        if hasattr(self, "_consumed_by") and self._consumed_by:
            # Redirect the method call to the new CompositeEnvelope
            return getattr(self._consumed_by, method.__name__)(*args, **kwargs)
        else:
            return method(self, *args, **kwargs)

    return wrapper


class CompositeEnvelope:
    def __init__(self, *envelopes):
        from photon_weave.state.envelope import Envelope

        self.envelopes = []
        self.states = []
        # If the state is consumed by another composite state, the reference is stored here
        self._consumed_by = None
        seen_composite_envelopes = set()
        for e in envelopes:
            if isinstance(e, CompositeEnvelope):
                if e not in seen_composite_envelopes:
                    self.envelopes.extend(e.envelopes)
                    self.states.extend(e.states)
                    seen_composite_envelopes.add(e)
            elif isinstance(e, Envelope):
                if (
                    not e.composite_envelope is None
                    and e.composite_envelope not in seen_composite_envelopes
                ):
                    self.states.extend(e.composite_envelope.states)
                    seen_composite_envelopes.add(e.composite_envelope)
                    self.envelopes.extend(e.composite_envelope.envelopes)
                else:
                    self.envelopes.append(e)
        for ce in seen_composite_envelopes:
            ce._consumed_by = self
            ce.states = []

        self.envelopes = list(set(self.envelopes))
        self.update_indices()
        for e in self.envelopes:
            e.composite_envelope = self

    @redirect_if_consumed
    def combine(self, *states):
        """
        Combines states into a product space
        """
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        # Check if states are already combined
        for _, obj_list in self.states:
            if all(target in obj_list for target in states):
                return

        for s in states:
            if not (isinstance(s, Fock) or isinstance(s, Polarization)):
                raise FockOrPolarizationExpectedException()
            included_states = []
            for env in self.envelopes:
                included_states.append(env.fock)
                included_states.append(env.polarization)
            if not any(s is state for state in included_states):
                raise StateNotInThisCompositeEnvelopeException()

        existing_product_states = []
        tmp = []
        for state in states:
            for i, s in enumerate(self.states):
                if state in s[1]:
                    existing_product_states.append(i)
                    tmp.append(state)

        new_product_states = [s for s in states if s not in tmp]

        # Combine
        if len(existing_product_states) > 0:
            target_ps = existing_product_states[0]
            existing_product_states.pop(0)
        else:
            target_ps = len(self.states)

        ## First combine existing spaces
        expected_expansion = max([s.expansion_level for s in states])
        if expected_expansion < ExpansionLevel.Vector:
            expected_expansion = ExpansionLevel.Vector

        ## Correct expansion in existing product spaces
        if target_ps != len(self.states):
            if self.states[target_ps][1][0].expansion_level < expected_expansion:
                self.states[target_ps][0] = np.outer(
                    self.states[target_ps][0].flatten(),
                    np.conj(self.states[target_ps][0].flatten()),
                )
        else:
            self.states.append([1, []])

        for i in existing_product_states:
            if self.states[i][1][0].expansion_level < expected_expansion:
                self.states[i][0] = np.outer(
                    self.states[i][0].flatten(),
                    np.conj(self.states[i][0].flatten()),
                )

        for eps in existing_product_states:
            existing = self.states[eps]
            self.states[target_ps][0] = np.kron(self.states[target_ps][0], existing[0])
            self.states[target_ps][1].extend(existing[1])
        ## Second combine new spaces
        for nps in new_product_states:
            while nps.expansion_level < expected_expansion:
                nps.expand()
            if nps.expansion_level == ExpansionLevel.Vector:
                if nps.state_vector is not None:
                    self.states[target_ps][0] = np.kron(
                        self.states[target_ps][0], nps.state_vector
                    )
                    nps.state_vector = None
                    self.states[target_ps][1].append(nps)
                else:
                    self.states[target_ps][0] = np.kron(
                        self.states[target_ps][0], nps.envelope.composite_vector
                    )
                    indices = [None, None]
                    indices[nps.envelope.fock.index] = nps.envelope.fock
                    indices[nps.envelope.polarization.index] = nps.envelope.polarization
                    nps.envelope.composite_vector = None
                    self.states[target_ps][1].append(nps)
            elif nps.expansion_level == ExpansionLevel.Matrix:
                if nps.density_matrix is not None:
                    self.states[target_ps][0] = np.kron(
                        self.states[target_ps][0], nps.density_matrix
                    )
                    nps.density_matrix = None
                    self.states[target_ps][1].append(nps)
                else:
                    self.states[target_ps][0] = np.kron(
                        self.states[target_ps][0], nps.envelope.composite_matrix
                    )
                    indices = [None, None]
                    indices[nps.envelope.fock.index] = nps.envelope.fock
                    indices[nps.envelope.polarization.index] = nps.envelope.polarization
                    self.states[target_ps][1].append(nps)
        # Delete old product spaces
        for index in sorted(existing_product_states, reverse=True):
            del self.states[index]
        self.update_indices()

    @redirect_if_consumed
    def update_indices(self):
        for major, _ in enumerate(self.states):
            for minor, state in enumerate(self.states[major][1]):
                state.set_index(minor, major)

    @redirect_if_consumed
    def add_envelope(self, envelope):
        self.envelopes.append(envelope)
        envelope.composite_envelope = self

    @redirect_if_consumed
    def expand(self, state):
        if state.envelope.expansion_level >= ExpansionLevel.Matrix:
            return
        state_index = None
        for i, s in enumerate(self.states):
            if state in s[1]:
                state_index = i
                break
        self.states[state_index][0] = np.outer(
            self.states[state_index][0].flatten(),
            np.conj(self.states[state_index][0].flatten()),
        )
        for s in self.states[state_index][1]:
            s.expansion_level = ExpansionLevel.Matrix

    @redirect_if_consumed
    def _find_composite_state_index(self, *states):
        composite_state_index = None
        for i, (_, states_group) in enumerate(self.states):
            if all(s in states_group for s in states):
                composite_state_index = i
                return composite_state_index
        return None

    @redirect_if_consumed
    def rearange(self, *ordered_states):
        """
        Uses the swap operation to rearange the states, according to the given order
        The ordered states must already be in the same product
        """
        composite_state_index = None
        for i, (_, states_group) in enumerate(self.states):
            if all(s in states_group for s in ordered_states):
                composite_state_index = i
                break

        if composite_state_index is None:
            raise ValueError("Specified states do not match any composite state.")

        # Check if the states are already in the desired order
        current_order = self.states[composite_state_index][1]
        if all(
            ordered_states[i] == current_order[i]
            for i in range(min(len(ordered_states), len(current_order)))
        ):
            return

        dimensions = [
            state.dimensions for state in self.states[composite_state_index][1]
        ]

        new_order = self.states[composite_state_index][1].copy()
        for idx, ordered_state in enumerate(ordered_states):
            if new_order.index(ordered_state) != idx:
                tmp = new_order[idx]
                old_idx = new_order.index(ordered_state)
                new_order[idx] = ordered_state
                new_order[old_idx] = tmp
        self._reorder_states(new_order, composite_state_index)

    @redirect_if_consumed
    def _reorder_states(self, order, state_index):
        # Calculate the total dimension of the composite system
        total_dim = np.prod([s.dimensions for s in self.states[state_index][1]])

        # Calculate the current (flattened) index for each subsystem
        current_dimensions = [s.dimensions for s in self.states[state_index][1]]
        target_dimensions = [s.dimensions for s in order]  # The new order's dimensions

        # Initialize the permutation matrix
        permutation_matrix = np.zeros((total_dim, total_dim))

        # Calculate new index for each element in the flattened composite state
        for idx in range(total_dim):
            # Determine the multi-dimensional index in the current order
            multi_idx = np.unravel_index(idx, current_dimensions)

            # Map the multi-dimensional index to the new order
            new_order_multi_idx = [
                multi_idx[self.states[state_index][1].index(s)] for s in order
            ]

            # Calculate the linear index in the new order
            new_idx = np.ravel_multi_index(new_order_multi_idx, target_dimensions)

            # Update the permutation matrix
            permutation_matrix[new_idx, idx] = 1

        if order[0].expansion_level == ExpansionLevel.Vector:
            self.states[state_index][0] = (
                permutation_matrix @ self.states[state_index][0]
            )
            self.states[state_index][1] = order
        elif order[0].expansion_level == ExpansionLevel.Matrix:
            self.states[state_index][0] = (
                permutation_matrix
                @ self.states[state_index][0]
                @ permutation_matrix.conj().T
            )
            self.states[state_index][1] = order

    @redirect_if_consumed
    def apply_operation(self, operation, *states):
        from photon_weave.operation.composite_operation import CompositeOperation
        from photon_weave.operation.fock_operation import FockOperation
        from photon_weave.operation.polarization_operations import PolarizationOperation
        from photon_weave.state.composite_envelope import CompositeEnvelope
        from photon_weave.state.envelope import Envelope

        csi = self._find_composite_state_index(states[0])
        if isinstance(operation, FockOperation) or isinstance(
            operation, PolarizationOperation
        ):
            if csi is None:
                states[0].apply_operation(operation)
            else:
                self._apply_operator(operation, *states)
        elif isinstance(operation, CompositeOperation):
            operation.operate(*states)

    @redirect_if_consumed
    def _apply_operator(self, operation, *states):
        """
        Assumes the spaces are correctly ordered
        """
        from photon_weave.operation.fock_operation import (
            FockOperation,
            FockOperationType,
        )
        from photon_weave.operation.polarization_operations import (
            PolarizationOperation,
            PolarizationOperationType,
        )
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        csi = self._find_composite_state_index(*states)
        composite_operator = 1
        skip_count = 0
        for i, state in enumerate(self.states[csi][1]):
            if skip_count > 0:
                skip_count -= 1
                continue
            if all(
                state is self.states[csi][1][i + j] for j, state in enumerate(states)
            ):
                if operation.operator is None:
                    operation.compute_operator(state.dimensions)
                composite_operator = np.kron(composite_operator, operation.operator)
                skip_count += len(states) - 1
            else:
                identity = None
                if isinstance(state, Fock):
                    identity = FockOperation(FockOperationType.Identity)
                    identity.compute_operator(state.dimensions)
                    identity = identity.operator
                elif isinstance(state, Polarization):
                    identity = PolarizationOperation(PolarizationOperationType.I)
                    identity.compute_operator()
                    identity = identity.operator
                composite_operator = np.kron(composite_operator, identity)
        if self.states[csi][1][0].expansion_level == ExpansionLevel.Vector:
            self.states[csi][0] = composite_operator @ self.states[csi][0]
        elif self.states[csi][1][0].expansion_level == ExpansionLevel.Matrix:
            self.states[csi][0] = composite_operator @ self.states[csi][0]
            self.states[csi][0] = self.states[csi][0] @ composite_operator.conj().T

    @redirect_if_consumed
    def measure(self, *states) -> int:
        """
        Measures the number state
        """
        from photon_weave.state.envelope import Envelope
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        outcomes = []
        nstates = []
        for s in states:
            if isinstance(s, Envelope):
                nstates.append(s.fock)
                nstates.append(s.polarization)
            elif isinstance(s, Polarization) or isinstance(s, Fock):
                nstates.append(s)

        for tmp, s in enumerate(nstates):
            if s.envelope not in self.envelopes:
                raise StateNotInThisCompositeEnvelopeException()
            if not isinstance(s.index, tuple):
                if isinstance(s, Polarization):
                    s.measure(remove_composite=False, partial=True)
                else:
                    outcomes.append(s.measure(remove_composite=False, partial=True))
            else:
                if isinstance(s, Polarization):
                    if s.expansion_level == ExpansionLevel.Vector:
                        self._measure_vector(s)
                    else:
                        self._measure_matrix(s)
                else:
                    if s.expansion_level == ExpansionLevel.Vector:
                        outcomes.append(self._measure_vector(s))
                    else:
                        outcomes.append(self._measure_matrix(s))
        for s in nstates:
            s.envelope.composite_envelope = None
            s.envelope._set_measured()
            if s.envelope in self.envelopes:
                self.envelopes.remove(s.envelope)
        self.update_indices()
        return outcomes

    @redirect_if_consumed
    def POVM_measurement(self, states, operators, non_destructive=False) -> int:
        from photon_weave.state.envelope import Envelope
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        self.combine(*states)
        self.rearange(*states)

        # Find the index of the composite state
        composite_state_index = self._find_composite_state_index(*states)
        if composite_state_index is None:
            raise ValueError("States are not all in the same composite state")

        composite_state = self.states[composite_state_index][0]
        state_dimensions = [
            state.dimensions for state in self.states[composite_state_index][1]
        ]

        # Create combuined operator
        total_dimensions = np.prod(state_dimensions)
        probabilities = []
        outcome_states = []
        target_index = self.states[composite_state_index][1].index(states[0])
        operators = [
            pad_operator(op, state_dimensions, target_index) for op in operators
        ]
        validate_povm_operators(operators, total_dimensions)
        for operator in operators:
            # Apply the padded operator depending on the state representation
            if (
                self.states[composite_state_index][1][0].expansion_level
                is ExpansionLevel.Vector
            ):
                outcome_state = operator @ composite_state
                prob = composite_state.T.conj() @ outcome_state
                prob = prob[0][0]
            else:  # Matrix state
                prob = np.trace(operator @ composite_state)
                outcome_state = operator @ composite_state @ operator.T.conj()
                if np.trace(outcome_state) > 0:
                    outcome_state = outcome_state / np.trace(outcome_state)
                else:
                    outcome_state = np.zeros_like(outcome_state)
            probabilities.append(prob)
            outcome_states.append(outcome_state)
        probabilities = np.real(np.array(probabilities))
        probabilities /= probabilities.sum()
        probabilities = np.round(probabilities, decimals=10)
        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        chosen_state = outcome_states[chosen_index]
        if not non_destructive:
            if (
                self.states[composite_state_index][1][0].expansion_level
                is ExpansionLevel.Vector
            ):
                self.states[composite_state_index][0] = chosen_state / np.linalg.norm(
                    chosen_state
                )
            else:
                self.states[composite_state_index][0] = chosen_state / np.trace(
                    chosen_state
                )
        return chosen_index

    def _measure_vector(self, state):
        o = None
        if not isinstance(state.index, tuple):
            return state.measure()
        s_idx, subs_idx = state.index

        dims = [s.dimensions for s in self.states[s_idx][1]]
        probabilities = []
        projection_states = []
        before = np.eye(int(np.prod(dims[:subs_idx])))
        after = np.eye(int(np.prod(dims[subs_idx + 1 :])))
        for i in range(state.dimensions):
            projection = np.zeros((state.dimensions, state.dimensions))
            projection[i, i] = 1
            full_projection = np.kron(np.kron(before, projection), after)
            projected_state = full_projection @ self.states[s_idx][0]
            prob = np.linalg.norm(projected_state) ** 2
            probabilities.append(prob)
            projection_states.append(projected_state)
        o = np.random.choice(range(state.dimensions), p=probabilities)
        self.states[s_idx][0] = projection_states[o]
        self.states[s_idx][0] /= np.linalg.norm(projection_states[o])
        self._trace_out(state)
        state._set_measured(remove_composite=False)
        return o

    def _measure_matrix(self, state):
        o = None
        if not isinstance(state.index, tuple):
            return state.measure()
        s_idx, subs_idx = state.index

        dims = [s.dimensions for s in self.states[s_idx][1]]
        probabilities = []
        projection_states = []
        before = np.eye(int(np.prod(dims[:subs_idx])))
        after = np.eye(int(np.prod(dims[subs_idx + 1 :])))
        rho = self.states[s_idx][0]
        for i in range(state.dimensions):
            projection = np.zeros((state.dimensions, state.dimensions))
            projection[i, i] = 1
            full_projection = np.kron(np.kron(before, projection), after)
            projected_state = full_projection @ rho @ full_projection.conj().T
            prob = np.trace(projected_state)
            probabilities.append(prob)
            projection_states.append(projected_state)
        o = np.random.choice(range(state.dimensions), p=probabilities)
        self.states[s_idx][0] = projection_states[o] / probabilities[o]
        self._trace_out(state)
        state._set_measured(remove_composite=False)
        return o

    def _trace_out(self, state, destructive=True):
        space_index, subsystem_index = state.index
        dims = [s.dimensions for s in self.states[space_index][1]]
        if len(self.states[space_index][1]) < 2:
            return
        if state.expansion_level < ExpansionLevel.Matrix:
            self.expand(state)
        n = len(dims)
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        rho = self.states[space_index][0]

        input_str = ""
        output_str = ""
        input_2_str = ""
        output_2_str = ""
        reshape_dims = (*dims, *dims)
        c_id = 0
        c_2_id = 0
        trace_letters = iter(letters)

        # Build einsum string to trace out the specific system
        for i in range(len(reshape_dims)):
            if i % len(dims) == subsystem_index:
                input_str += next(trace_letters)
            else:
                char = next(trace_letters)
                input_str += char
                output_str += char

        trace_letters = iter(letters)

        for i in range(len(reshape_dims)):
            if i % len(dims) == subsystem_index:
                char = next(trace_letters)
                input_2_str += char
                output_2_str += char
            else:
                input_2_str += next(trace_letters)
        einsum_str = f"{input_str}->{output_str}"
        einsum_2_str = f"{input_2_str}->{output_2_str}"

        rho = rho.reshape(reshape_dims)
        traced_out_state = np.einsum(einsum_2_str, rho)
        traced_out_state = traced_out_state / np.trace(traced_out_state)
        rho = np.einsum(einsum_str, rho)
        new_dims = np.prod(dims) // state.dimensions
        rho = rho.reshape(new_dims, new_dims)
        if destructive:
            self.states[space_index][0] = rho

            self.contract(state)
            # Update system information post trace-out
            del self.states[space_index][1][subsystem_index]
        return traced_out_state

    @redirect_if_consumed
    def contract(self, state):
        if state.expansion_level < ExpansionLevel.Matrix:
            return
        space_index = state.index[0]
        # Assuming self.states[space_index] correctly references the density matrix
        rho = self.states[space_index][0]
        # Square the density matrix before taking the trace
        tr_rho_squared = np.trace(np.dot(rho, rho))
        if not np.isclose(tr_rho_squared, 1.0):
            return
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        # Assuming you want to keep the eigenvector corresponding to the largest eigenvalue
        # And assuming the structure of self.states allows direct replacement
        vector = eigenvectors[:, np.argmax(eigenvalues)].reshape(-1, 1)
        # Update the state with the column vector
        self.states[space_index][0] = vector

        # This line seems to attempt to update expansion_level for multiple states,
        # Ensure the structure of self.states[space_index][1] (if it exists) supports iteration like this
        for s in self.states[space_index][1]:
            s.expansion_level = ExpansionLevel.Vector


def pad_operator(operator, state_dimensions, target_index):
    """
    Expands an operator to act on the full Hilbert space of a composite system, targeting a specific subsystem
    Args:
    operator (np.ndarray): The operator that acts on the target state.
    state_dimensions (list): A list of dimensions for each part of the composite system
    target_index (int): The index of the state within the composite system where the operator should be applied

    Returns:
    np.ndarray: An operator padded to act actoss the entire Hilbert space.
    """

    padded_operator = np.eye(1)
    span = 0
    operator_dim = operator.shape[0]
    cumulative_dim = 1
    i = 0

    while cumulative_dim < operator_dim:
        if target_index + span == len(state_dimensions):
            raise ValueError(
                "Operator dimensions exceed available system dimensions from target index"
            )
        cumulative_dim *= state_dimensions[target_index + span]
        span += 1
    if cumulative_dim != operator_dim:
        raise ValueError(
            "Operator dimensions do not match the dimensions of the spanned subsystems."
        )

    for index, dim in enumerate(state_dimensions):
        if index < target_index or index >= target_index + span:
            padded_operator = np.kron(padded_operator, np.eye(dim))
        elif index == target_index:
            padded_operator = np.kron(padded_operator, operator)

    return padded_operator


def validate_povm_operators(operators, dimension):
    """Ensure that the sum of operators equals the identity matrix of the given dimension."""
    sum_operators = sum(operators)
    identity = np.eye(dimension)
    if not np.allclose(sum_operators, identity):
        raise ValueError(
            "Provided POVM operators do not sum up to the identity operator."
        )
