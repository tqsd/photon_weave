from contextlib import ExitStack
import numpy as np
import itertools
import jax
import jax.numpy as jnp
import itertools
import uuid
from typing import Union, List, Optional, Dict
import threading
from dataclasses import dataclass, field, InitVar

from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.photon_weave import Config


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

@dataclass(slots=True)
class ProductState:
    """
    Stores Product state and references to its constituents
    """
    state: jnp.ndarray = field(default_factory=jnp.ndarray)
    expansion_level: ExpansionLevel = field(default_factory=ExpansionLevel)
    state_objs: List[Union['Fock', 'Polarization']] = field(default_factory=list)
    container: 'CompositeEnvelopeContainer' = field(default_factory=lambda: CompositeEnvelopeContainer)

    def expand(self) -> None:
        """
        Expands the state from vector to matrix
        """
        if self.expansion_level < ExpansionLevel.Matrix:
            self.state = jnp.outer(
                self.state.flatten(),
                jnp.conj(self.state.flatten())
            )
            self.expansion_level = ExpansionLevel.Matrix
            for state in self.state_objs:
                state.expansion_level = ExpansionLevel.Matrix

    def contract(self, tol:float) -> None:
        """
        Attempts to contract the representation from matrix to vector

        Parameters
        ----------
        tol: float
            tolerance when comparing trace
        """
        if self.expansion_level is ExpansionLevel.Matrix:
            # If the state is mixed, return
            if jnp.abs(jnp.trace(jnp.matmul(
                    self.state, self.state
            ))-1) >= tol:
                return
            eigenvalues, eigenvectors = jnp.linalg.eigh(self.density_matrix)
            pure_state_index = jnp.argmax(jnp.abs(eigenvalues - 1.0) < tol)
            assert pure_state_index is not None, "Pure state indx should not be None"
            self.state = eigenvectors[:, pure_state_index].reshape(-1,1)
            for state in self.state_objs:
                state.expansion_level = ExpansionLevel.Vector

    def reorder(self, *ordered_states) -> None:
        """
        Changes the order of tensoring, all ordered states need to be given
        """
        assert all(so in ordered_states for so in self.state_objs), "All state objects need to be given"
        old_dims = [os.dimensions for os in ordered_states]
        if self.expansion_level == ExpansionLevel.Vector:
            shape = [so.dimensions for so in self.state_objs]
            state = self.state.reshape(shape)
            new_order = [self.state_objs.index(so) for so in ordered_states]
            state = jnp.transpose(state, axes=new_order)
            self.state = state.reshape(-1,1)
            self.state_objs = ordered_states
        elif self.expansion_level == ExpansionLevel.Matrix:
            shape = [os.dimensions for os in ordered_states]*2
            state = self.state.reshape(shape)
            c1 = itertools.count(start=0)
            old_order = list(range(len(shape)))
            order_first = {s:self.state_objs.index(s) for s in ordered_states}
            order_second= {s:self.state_objs.index(s)+len(self.state_objs) for s in ordered_states}
            new_order = [order_first[s] for s in ordered_states]
            new_order.extend([order_second[s] for s in ordered_states])
            state = jnp.transpose(state, axes=new_order)
            self.state = state.reshape(sum([s.dimensions for s in ordered_states]), sum([s.dimensions for s in ordered_states]))
            self.state_objs = ordered_states

        self.container.update_all_indices()

    def measure(self, *states:Optional[List[Union['Fock', 'Polarization']]]) -> Dict[Union['Fock', 'Polarization'],int]:
        """
        Measure given states in this product space
        """

        assert all(so in self.state_objs for so in states), "All state objects need to be in product state"
        outcomes = {}
        C = Config()
        remaining_states = [s for s in self.state_objs]

        # Also include the envelope other states
        other_states = []
        for s in states:
            if s.envelope is not None:
                if s is s.envelope.fock:
                    other_states.append(s.envelope.polarization)
                elif s is s.envelope.polarization:
                    other_states.append(s.envelope.fock)
        other_states_in_product_space = [s for s in other_states if s in self.state_objs]
        state_list = [s for s in states]
        state_list.extend(other_states_in_product_space)

        remove_states = [s for s in state_list]

        if self.expansion_level == ExpansionLevel.Vector:
            shape = [so.dimensions for so in self.state_objs]
            shape.append(1)
            ps = self.state.reshape(shape)
            for idx, state in enumerate(state_list):
                # Constructing the einsum str
                counter = itertools.count(start=0)
                einsum = [[],[]]
                for so in remaining_states:
                    c = next(counter)
                    if so is state:
                        einsum[1].append(c)
                    einsum[0].append(c)
                c = next(counter)
                einsum[0].append(c)
                einsum[1].append(c)
                einsum = [[chr(97+x) for x in ep] for ep in einsum]
                einsum = ["".join(x) for x in einsum]
                einsum = "->".join(einsum)

                # Project the state with einsum
                projected_state = jnp.einsum(einsum, ps)

                # Outcome Probabilities
                probabilities = jnp.abs(projected_state.flatten())**2
                probabilities /= jnp.sum(probabilities)

                key = C.random_key
                outcomes[state] = int(jax.random.choice(
                    key,
                    a=jnp.array(list(range(state.dimensions))),
                    p=probabilities
                ))
                indices = [slice(None)]*len(ps.shape)
                indices[remaining_states.index(state)] = outcomes[state]
                ps = ps[tuple(indices)]
                remaining_states.remove(state)
                state._set_measured()
                self.state_objs.remove(state)
            if len(self.state_objs) > 0:
                outcome_dims = sum([obj.dimensions for obj in self.state_objs if obj not in states])
                self.state = ps.reshape(-1,1)
                self.state /= jnp.linalg.norm(self.state)
            else:
                self.state = jnp.array([[1]])
        elif self.expansion_level == ExpansionLevel.Matrix:
            shape = [so.dimensions for so in self.state_objs]*2
            ps = self.state.reshape(shape)
            for idx, state in enumerate(state_list):
                # Constructing the einsum str
                counter = itertools.count(start=0)
                einsum = [[],[]]
                for _ in range(2):
                    for so in remaining_states:
                        c = next(counter)
                        if so is state:
                            einsum[1].append(c)
                        einsum[0].append(c)
                einsum = [[chr(97+x) for x in ep] for ep in einsum]
                einsum = ["".join(x) for x in einsum]
                einsum = "->".join(einsum)

                # Project the state with einsum
                projected_state = jnp.einsum(einsum, ps)

                # Outcome Probabilities
                probabilities = jnp.abs(jnp.diag(projected_state))
                probabilities /= sum(probabilities)

                key = C.random_key
                outcomes[state] = int(jax.random.choice(
                    key,
                    a=jnp.array(list(range(state.dimensions))),
                    p=probabilities
                ))

                state_index = remaining_states.index(state)
                row_idx = state_index
                col_idx = state_index + len(remaining_states)
                indices = [slice(None)]*len(ps.shape)
                indices[row_idx] = outcomes[state]
                indices[col_idx] = outcomes[state]
                ps = ps[tuple(indices)]
                remaining_states.remove(state)
                state._set_measured()
                self.state_objs.remove(state)
            if len(self.state_objs) > 0:
                outcome_dims = sum([obj.dimensions for obj in self.state_objs if obj not in states])
                ps = ps.flatten()
                num_elements = ps.size
                sqrt = int(jnp.ceil(jnp.sqrt(num_elements)))
                self.state = ps.reshape((sqrt,sqrt))
                self.state /= jnp.linalg.norm(self.state)
            else:
                self.state = jnp.array([[1]])


        for os in other_states:
            if os.measured:
                continue
            if os not in self.state_objs:
                if isinstance(os.index, tuple):
                    outcome = os.envelope.composite_envelope.measure(os)
                else: 
                    outcome = os.measure()
                for s, out in outcome.items():
                    outcomes[s] = out
        # Remove the state objs from the states 
        for so in remove_states:
            so._set_measured()
            if so.envelope is not None:
                so.envelope._set_measured()
            if so in self.state_objs:
                # This can be skipped in coverage
                # States actually get removed immediately when measured
                self.state_objs.remove(so) # pragma: no cover
        return outcomes

    def measure_POVM(self, states, operators) -> list[int]:
        pass

    def apply_kraus(self, states, operators) -> None:
        pass

    @property
    def is_empty(self) -> bool:
        if len(self.state_objs) == 0:
            return True
        return False



@dataclass(slots=True)
class CompositeEnvelopeContainer:
    envelopes: List['Envelope'] = field(default_factory=list)
    states: List[Optional[ProductState]] = field(default_factory=list)
    locks: List[Optional[threading.Lock]] = field(init=False, default_factory=list)
    state_lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self):
        for state in self.states:
            if isinstance(state, jnp.ndarray):
                self.locks.append(threading.Lock())

    def add_state(self, state: jnp.ndarray):
        """
        Adds the state and creates lock for it

        Parameters
        ----------
        state: jnp.ndarray
            State which should be added
        """
        with self.state_lock:
            self.states.append(state)
            self.locks.append(threading.Lock())

    def lock_state(self, index:int):
        """
        Acquire lock for the state at specified index
        """
        if self.locks[index] is not None:
            self.locks[index].acquire()

    def release_state(self, index:int):
        """
        Release lock for the state at the specific index
        """
        if self.lock[index] is not None:
            self.locks[index].release()

    def append_states(self, other: 'CompositeEnvelopeContainer'):
        """
        Appnds the states of two composite envelope containers
        Parameters
        ----------
        other: CompositeEnvelopeContainer
            Other composite envelope container 
        """
        assert isinstance(other, CompositeEnvelopeContainer)
        with ExitStack() as stack:
            stack.enter_context(self.state_lock)
            stack.enter_context(other.state_lock)
            for lock in [*self.locks, *other.locks]:
                if lock is not None:
                    stack.enter_context(lock)
            self.states.extend(other.states)
            self.envelopes.extend(other.envelopes)
            self.locks.extend(other.locks)

    def remove_empty_product_states(self) -> None:
        """
        Checks if a product state is empty and if so
        removes it
        """
        for state in self.states:
            if state.is_empty:
                self.states.remove(state)

    def update_all_indices(self) -> None:
        """
        Updates all of the indices of the state_objs
        Note: Might not be necessary
        """
        for state_index, state in enumerate(self.states):
            for i,so in enumerate(state.state_objs):
                so.extract((state_index, i))


class CompositeEnvelope:
    """
    Composite Envelope is a pointer to a container, which includes the state
    Multiple Composite enveopes can point to the same containers.
    """
    _containers = {}
    _instances = {}

    def __init__(self, *envelopes: Optional[List[Union['CompositeEnvelope', 'Envelope']]]) -> 'CompositeEnvelope':
        from photon_weave.state.envelope import Envelope
        self.uid = uuid.uuid4()
        # Check if there are composite envelopes in the argument list
        composite_envelopes = [e for e in envelopes if isinstance(e, CompositeEnvelope)]
        envelopes = [e for e in envelopes if isinstance(e, Envelope)]
        for e in envelopes:
            if (e.composite_envelope is not None and
                e.composite_envelope not in composite_envelopes):
                composite_envelopes.append(e.composite_envelope)

        ce_container = None
        for ce in composite_envelopes:
            if ce_container is None:
                ce_container = CompositeEnvelope._containers[ce.uid]
            else:
                ce_container.append_states(CompositeEnvelope._containers[ce.uid])
            ce.uid = self.uid
        if ce_container is None:
            ce_container = CompositeEnvelopeContainer()
        for e in envelopes:
            if e not in ce_container.envelopes:
                ce_container.envelopes.append(e)
        CompositeEnvelope._containers[self.uid] = ce_container
        if not CompositeEnvelope._instances.get(self.uid):
            CompositeEnvelope._instances[self.uid] = []
        CompositeEnvelope._instances[self.uid].append(self)
        self.update_composite_envelope_pointers()

    def __repr__(self) -> str:
        return f"CompositeEnvelope(uid={self.uid}, envelopes={[e.uid for e in self.envelopes]}, state_objects={[s.uid for s in self.state_objs]})"

    @property
    def envelopes(self):
        return CompositeEnvelope._containers[self.uid].envelopes

    @property
    def state_objs(self) -> List[Union['Fock', 'Polarization']]:
        state_objs = []
        for e in self.envelopes:
            state_objs.extend([e.fock, e.polarization])
        return state_objs

    @property
    def product_states(self) -> List[Optional[ProductState]]:
        return CompositeEnvelope._containers[self.uid].states
        
    @property
    def container(self) -> CompositeEnvelopeContainer:
        return CompositeEnvelope._containers[self.uid]

    @property
    def states(self) -> List[Optional[Union['Fock', 'Polarization']]]:
        return CompositeEnvelope._containers[self.uid].states

    def update_composite_envelope_pointers(self) -> None:
        """
        Updates all the envelopes to point to this composite envelope
        """
        for envelope in self.envelopes:
            envelope.set_composite_envelope_id(self.uid)

    def combine(self, *state_objs: Union['Fock', 'Polarization']) -> None:
        """
        Combines given states into a product state.

        Parameters
        ----------
        state_objs: Union['Fock', 'Polarization']
           Accepts many state_objs
        """
        from photon_weave.state.polarization import Polarization
        from photon_weave.state.fock import Fock

        # Check for the types
        for so in state_objs:
            assert isinstance(so, Fock) or isinstance(so, Polarization), f"got {type(so)}, expected {t}"

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
            for i, ps in enumerate(self.product_states):
                if state in ps.state_objs:
                    existing_product_states.append(ps)
        """
        Ensure all states have the same expansion levels
        """
        minimum_expansion_level = ExpansionLevel.Vector
        for obj in state_objs:
            if obj.expansion_level > minimum_expansion_level:
                minimum_expansion_level = ExpansionLevel.Matrix
                break

        # Expand the product spaces
        for product_state in existing_product_states:
            while product_state.expansion_level < minimum_expansion_level:
                product_state.expand()

        for obj in state_objs:
            if obj.index is None:
                while obj.expansion_level < minimum_expansion_level:
                    obj.expand()
            elif isinstance(obj.index, int):
                while obj.expansion_level < minimum_expansion_level:
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
                state_vector_or_matrix,
                product_state.state)
            state_order.extend(product_state.state_objs)
            product_state.state_objs = []
        for so in target_state_objs:
            if so.envelope is not None and so.index is not None and not isinstance(so.index, tuple):
                if minimum_expansion_level is ExpansionLevel.Vector:
                    state_vector_or_matrix = jnp.kron(
                        state_vector_or_matrix, so.envelope.composite_vector
                    )
                else:
                    state_vector_or_matrix = jnp.kron(
                        state_vector_or_matrix, so.envelope.composite_matrix
                    )
                indices = [None, None]
                indices[so.envelope.fock.index] = so.envelope.fock
                indices[so.envelope.polarization.index] = so.envelope.polarization
                state_order.extend(indices)
            if so.index is None:
                if minimum_expansion_level is ExpansionLevel.Vector:
                    state_vector_or_matrix = jnp.kron(
                        state_vector_or_matrix, so.state_vector
                    )
                else:
                    state_vector_or_matrix = jnp.kron(
                        state_vector_or_matrix, so.density_matrix
                    )
                state_order.append(so)

        """
        Create a new product state object and append it to the states
        """
        ps = ProductState(
            expansion_level = minimum_expansion_level,
            state = state_vector_or_matrix,
            state_objs = state_order,
            container = self._containers[self.uid]

        )

        CompositeEnvelope._containers[self.uid].states.append(ps)

        """
        Remove empty states
        """
        self.container.remove_empty_product_states()
        self.container.update_all_indices()

    def reorder(self, *ordered_states: List[Optional[Union["Fock", "Polarization"]]]) -> None:
        """
        Changes the order of the states in the produce space
        If not all states are given, the given states will be
        put in the given order at the beginnig of the product
        states

        Parameters
        ----------
        ordered_states: List[Optional[Fock, Polarization]]
            ordered list of states
        """

        # Check if given states are shared in a product space
        states_are_combined = False
        for ps in self.states:
            if all (s in ps.state_objs for s in ordered_states):
                states_are_combined
        if not states_are_combined:
            self.combine(*ordered_states)

        # Get the correct product state:
        ps = [p for p in self.states if all(so in p.state_objs for so in ordered_states)][0]

        # Create order
        new_order = ps.state_objs.copy()
        for i, ordered_state in enumerate(ordered_states):
            if new_order.index(ordered_state) != i:
                tmp = new_order[i]
                old_idx = new_order.index(ordered_state)
                new_order[i] = ordered_state
                new_order[old_idx] = tmp
        ps.reorder(*new_order)

    def measure(self, *states: List[Union['Fock', 'Polarization']]) -> List[int]:
        product_states = [p for p in self.states if any(so in p.state_objs for so in states)]
        outcomes = {}
        for ps in product_states:
            ps_states = [so for so in states if so in ps.state_objs]
            out = ps.measure(*ps_states)
            for key, item in out.items():
                outcomes[key] = item
        self._containers[self.uid].update_all_indices()
        self._containers[self.uid].remove_empty_product_states()
        return outcomes



class OldCompositeEnvelope:
    __slots__ = ("envelopes", "states", "_consumed_by", "__dict__", "_composite_envelope_pairs")

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
                    e.composite_envelope is not None
                    and e.composite_envelope not in seen_composite_envelopes
                ):
                    self.states.extend(e.composite_envelope.states)
                    seen_composite_envelopes.add(e.composite_envelope)
                    self.envelopes.extend(e.composite_envelope.envelopes)
                    self._update_other_composite_envelope_reference(
                        e.composite_envelope
                    )
                    ce = e.composite_envelope
                    if ce not in self._composite_envelope_equals:
                        self._composite_envelope_equals.append(ce)
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
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        rho = self.states[space_index][0]

        input_str = ""
        output_str = ""
        input_2_str = ""
        output_2_str = ""
        reshape_dims = (*dims, *dims)

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
