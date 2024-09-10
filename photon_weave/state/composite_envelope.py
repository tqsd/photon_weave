from contextlib import ExitStack
import numpy as np
import itertools
import jax
import jax.numpy as jnp
import uuid
from typing import Union, List, Optional, Dict, Tuple, TYPE_CHECKING, cast, Callable
import threading
from dataclasses import dataclass, field, InitVar

from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.photon_weave import Config
from photon_weave._math.ops import kraus_identity_check

# For static type checks
if TYPE_CHECKING:
    from photon_weave.state.envelope import Envelope # pragma: no cover
    from photon_weave.state.fock import Fock # pragma: no cover
    from photon_weave.state.polarization import Polarization # pragma: no cover
    from photon_weave.state.base_state import BaseState # pragma: no cover

@dataclass(slots=True)
class ProductState:
    """
    Stores Product state and references to its constituents
    """
    expansion_level: ExpansionLevel
    container: 'CompositeEnvelopeContainer'
    uid: uuid.UUID = field(default_factory=uuid.uuid4)
    state: jnp.ndarray = field(default_factory=lambda: jnp.array([[1]]))
    state_objs: List['BaseState'] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.uid)

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

    def contract(self, tol:float=1e-6) -> None:
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

            eigenvalues, eigenvectors = jnp.linalg.eigh(self.state)
            pure_state_index = jnp.argmax(jnp.abs(eigenvalues - 1.0) < tol)
            assert pure_state_index is not None, "Pure state indx should not be None"
            self.state = eigenvectors[:, pure_state_index].reshape(-1,1)
            for state in self.state_objs:
                state.expansion_level = ExpansionLevel.Vector
            self.expansion_level = ExpansionLevel.Vector

    def reorder(self, *ordered_states:'BaseState') -> None:
        """
        Changes the order of tensoring, all ordered states need to be given

        Parameters
        ----------
        *ordered_states: 'BaseState'
            States ordered in the new order
        """
        from photon_weave.state.base_state import BaseState
        assert all(so in ordered_states for so in self.state_objs), "All state objects need to be given"
        old_dims = [os.dimensions for os in ordered_states]
        if self.expansion_level == ExpansionLevel.Vector:
            shape = [so.dimensions for so in self.state_objs]
            state = self.state.reshape(shape)
            new_order = [self.state_objs.index(so) for so in ordered_states]
            state = jnp.transpose(state, axes=new_order)
            self.state = state.reshape(-1,1)
            self.state_objs = list(ordered_states)
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
            new_dims = jnp.prod(jnp.array([s.dimensions for s in ordered_states]))
            self.state = state.reshape((new_dims, new_dims))
            self.state_objs = list(ordered_states)

        self.container.update_all_indices()

    def measure(self, *states:'BaseState') -> Dict['BaseState',int]:
        """
        Measure given states in this product space

        Parameters
        ----------
        *states: BaseState
            States that should be measured

        Returns
        -------
        Dict[BaseState, int]
           Dictionary of outcomes
        """
        assert all(so in self.state_objs for so in states), "All state objects need to be in product state"
        outcomes = {}
        C = Config()
        remaining_states = [s for s in self.state_objs]

        # Also include the envelope other states
        other_states = []
        for s in states:
            if hasattr(s,"envelope"):
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
                einsum_list: List[List[int]] = [[],[]]
                for so in remaining_states:
                    c = next(counter)
                    if so is state:
                        einsum_list[1].append(c)
                    einsum_list[0].append(c)
                c = next(counter)
                einsum_list[0].append(c)
                einsum_list[1].append(c)
                einsum_list_str = [[chr(97+x) for x in ep] for ep in einsum_list]
                einsum_tmp = ["".join(x) for x in einsum_list_str]
                einsum = "->".join(einsum_tmp)
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
                indices: List[Union[slice, int]] = [slice(None)]*len(ps.shape)
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
                einsum_list = [[],[]]
                for _ in range(2):
                    for so in remaining_states:
                        c = next(counter)
                        if so is state:
                            einsum_list[1].append(c)
                        einsum_list[0].append(c)
                einsum_list_str = [[chr(97+x) for x in ep] for ep in einsum_list]
                einsum_str = ["".join(x) for x in einsum_list_str]
                einsum = "->".join(einsum_str)

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
                indices: List[Union[slice, int]] = [slice(None)]*len(ps.shape)
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
            if hasattr(so, 'envelope') and so.envelope is not None:
                so.envelope._set_measured()
            if so in self.state_objs:
                # This can be skipped in coverage
                # States actually get removed immediately when measured
                self.state_objs.remove(so) # pragma: no cover

        # Contract if set so
        if C.contractions:
            self.contract()
        return outcomes

    def measure_POVM(self, operators:List[Union[np.ndarray, jnp.ndarray]],
                     *states:'BaseState',
                     destructive: bool = True,
                     ) -> Tuple[int, Dict['BaseState', int]]:
        """
        Perform a POVM measurement.

        Parameters
        ----------
        operators: List[Union[np.ndarray, jnp.ndarray]]
            List of POVM operators
        *states: BaseState
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
        transpose_pattern = lambda x: [item for i in range(len(x)) for item in [i,i+len(x)]]
        operators = [op.reshape(op_shape).transpose(*transpose_pattern(states)) for op in operators]

        # Reshape the state
        ps = self.state.reshape([s.dimensions for s in self.state_objs]*2).transpose(
            *transpose_pattern(self.state_objs)
        )

        # Construct Einstein Sum String
        einsum_list:List[List[int]] = [[],[],[],[]]
        einsum_states:Dict['BaseState',List[int]] = {s:[] for s in self.state_objs}
        c1 = itertools.count(start=0)

        # Marking down the state indices
        for idx, s in enumerate(self.state_objs):
            for i in range(2):
                c = next(c1)
                einsum_states[s].append(c)
                einsum_list[1].append(c)
                einsum_list[3].append(c)

        # Marking down the first operator indices
        for i, s in enumerate(states):
            c = next(c1)
            einsum_list[0].append(c)
            einsum_list[0].append(einsum_states[s][0])
            einsum_list[3][i*len(states)] = c

        # Marking the second operator indices:
        for i,s in enumerate(states):
            c = next(c1)
            einsum_list[2].append(c)
            einsum_list[2].append(einsum_states[s][1])
            einsum_list[3][i*len(states)+1] = c

        # Assembling the einstein sum
        einsum_tmp= [ "".join([chr(97+i) for i in s]) for s in einsum_list]
        einsum_str = f"{einsum_tmp[0]},{einsum_tmp[1]},{einsum_tmp[2]}->{einsum_tmp[3]}"
        
        dims = jnp.prod(jnp.array([s.dimensions for s in self.state_objs]))
        # Get the probabilities
        prob_list: List[float] =[]
        for op in operators:
            prob_state = jnp.einsum(einsum_str, op, ps, jnp.conj(op)).transpose(
                *transpose_pattern(self.state_objs)).reshape((dims, dims))
            prob_list.append(float(jnp.trace(prob_state)))

        # Normalize the probabilities
        probabilities = jnp.array(prob_list)
        probabilities /= jnp.sum(probabilities)

        C = Config()
        key = C.random_key
        outcome = int(jax.random.choice(
            key,
            a=jnp.array(list(range(len(operators)))),
            p=jnp.array(probabilities)
        ))

        # Construct Post Measurement state
        new_dims = jnp.prod(jnp.array([s.dimensions for s in self.state_objs]))
        ps = jnp.einsum(einsum_str, operators[outcome], ps, jnp.conj(operators[outcome])).transpose(
            *transpose_pattern(self.state_objs)).reshape((new_dims, new_dims))
        self.state = ps / jnp.trace(ps)
        other_outcomes = {}
        if destructive:
            # Get correct Composite Envelope
            if isinstance(CompositeEnvelope._instances[self.container.composite_uid], list):
                other_outcomes = CompositeEnvelope._instances[self.container.composite_uid][0].measure(
                    *states
                )
                for s in states:
                    del other_outcomes[s]
        if C.contractions:
            self.contract()
        return (outcome, other_outcomes)

    def apply_kraus(self, operators:List[Union[np.ndarray, jnp.ndarray]],
                    *states:'BaseState') -> None:
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
            K = jnp.array(operators)
            shape = [s.dimensions for s in self.state_objs] + [1]
            op_shape = [s.dimensions for s in states] * 2

            # Computing transpose pattern
            transpose_pattern = []
            for i in range(len(states)):
                transpose_pattern.append(i)
                transpose_pattern.append(i+len(states))
                
            operators = [op.reshape(op_shape).transpose(*transpose_pattern) for op in operators]
            ps = self.state.reshape(shape)
            # Constructing einsum string
            einsum_list_list:List[List[int]] = [[],[],[]]
            c1 = itertools.count(start=0)

            einsum_states = {}
            for s in self.state_objs:
                c = next(c1)
                einsum_states[s] = [c]
                einsum_list_list[1].append(c)
            ed = next(c1)
            einsum_list_list[1].append(ed)
            for s in states:
                c = next(c1)
                einsum_list_list[0].append(c)
                einsum_states[s].append(c)
                einsum_list_list[0].append(einsum_states[s][0])

            for s in self.state_objs:
                if len(einsum_states[s])>1:
                    einsum_list_list[2].append(einsum_states[s][1])
                else:
                    einsum_list_list[2].append(einsum_states[s][0])
            einsum_list_list[2].append(ed)

            einsum_list_str:List[str] = [ "".join([chr(97+i) for i in s]) for s in einsum_list_list]
            einsum_str = f"{einsum_list_str[0]},{einsum_list_str[1]}->{einsum_list_str[2]}"
            for op in operators:
                ps = jnp.einsum(einsum_str, op, ps)
            self.state = ps.flatten().reshape(-1,1)
        elif self.expansion_level == ExpansionLevel.Matrix:

            op_shape = [s.dimensions for s in states]*2

            tp:Callable[[Tuple['BaseState',...]], List[int]] = lambda x: [item for i in range(len(x)) for item in [i,i+len(x)]]

            #transpose_pattern = []
            #for i in range(len(states)):
            #    transpose_pattern.append(i)
            #    transpose_pattern.append(i+len(states))

            ps = self.state.reshape([s.dimensions for s in self.state_objs]*2)
            operators = [op.reshape(op_shape).transpose(*tp(states)) for op in operators]
            # Constructing einsum string
            einsum_list_list= [[],[],[],[]]
            einsum_states = {s:[] for s in self.state_objs}
            c1 = itertools.count(start=0)

            # Marking down the state indices
            for idx, s in enumerate(self.state_objs):
                for i in range(2):
                    c = next(c1)
                    einsum_states[s].append(c)
                    einsum_list_list[1].append(c)
                    einsum_list_list[3].append(c)

            # Marking down the first operator indices
            for i, s in enumerate(states):
                c = next(c1)
                einsum_list_list[0].append(c)
                einsum_list_list[0].append(einsum_states[s][0])
                einsum_list_list[3][i*len(states)] = c

            # Marking the second operator indices:
            for i,s in enumerate(states):
                c = next(c1)
                einsum_list_list[2].append(c)
                einsum_list_list[2].append(einsum_states[s][1])
                einsum_list_list[3][i*len(states)+1] = c

            # Assembling the einstein sum
            einsum_list= [ "".join([chr(97+i) for i in s]) for s in einsum_list_list]
            einsum_str = f"{einsum_list[0]},{einsum_list[1]},{einsum_list[2]}->{einsum_list[3]}"

            for op in operators:
                ps = jnp.einsum(einsum_str, op, ps, jnp.conj(op))
            dims = sum([s.dimensions for s in self.state_objs])

            # Change the state back to the matrix form
            #transpose_pattern = get_transpose_pattern(self.state_objs)
            ps = ps.transpose(*tp(tuple(self.state_objs)))
            self.state = ps.reshape(dims, dims)
            C = Config()
            if C.contractions:
                self.contract()


    def trace_out(self, *states: 'BaseState') -> jnp.ndarray:
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
            # Reshape the system
            shape = [s.dimensions for s in self.state_objs] + [1]
            ps = self.state.reshape(shape)

            # Compute einsum string
            einsum_list_list:List[List[int]] = [[],[]]
            c1 = itertools.count(start=0)
            for so in self.state_objs:
                if not so in states:
                    c = next(c1)
                    einsum_list_list[0].append(c)
                else:
                    c = next(c1)
                    einsum_list_list[0].append(c)
                    einsum_list_list[1].append(c)
            c = next(c1)
            einsum_list_list[0].append(c)
            einsum_list_list[1].append(c)
            einsum_list = [[chr(97+x) for x in string] for string in einsum_list_list]
            einsum_str = ["".join(s) for s in einsum_list]
            einsum = f"{einsum_str[0]}->{einsum_str[1]}"


            traced_out_state = jnp.einsum(einsum, ps)
            traced_out_state = traced_out_state.reshape((-1, 1))
            return traced_out_state
        elif self.expansion_level == ExpansionLevel.Matrix:
            transpose_pattern = lambda x: [item for i in range(len(x)) for item in [i,i+len(x)]]

            # Reshape the state
            ps = self.state.reshape([s.dimensions for s in self.state_objs]*2).transpose(
                *transpose_pattern(self.state_objs)
            )

            # Compute einstein sum string
            einsum_list_list = [[],[]]
            c1 = itertools.count(start=0)
            for so in self.state_objs:
                if not so in states:
                    c = next(c1)
                    einsum_list_list[0].extend([c,c])
                else:
                    cl = [next(c1), next(c1)]
                    einsum_list_list[0].extend(cl)
                    einsum_list_list[1].extend(cl)

            einsum_list = [[chr(97+x) for x in string] for string in einsum_list_list]
            einsum_str = ["".join(s) for s in einsum_list]
            einsum = f"{einsum_str[0]}->{einsum_str[1]}"

            new_dims = jnp.prod(jnp.array([s.dimensions for s in states]))

            traced_out_state = jnp.einsum(einsum, ps)
            traced_out_state = traced_out_state.transpose(transpose_pattern(states)).reshape(
                (new_dims, new_dims))
            return traced_out_state
        else:
            raise ValueError("Something went wrong") #pragma: no cover


    @property
    def is_empty(self) -> bool:
        if len(self.state_objs) == 0:
            return True
        return False



@dataclass(slots=True)
class CompositeEnvelopeContainer:
    composite_uid: uuid.UUID
    envelopes: List['Envelope'] = field(default_factory=list)
    states: List[ProductState] = field(default_factory=list)

    def append_states(self, other: 'CompositeEnvelopeContainer') -> None:
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
        Note: Might not be necessary
        """
        for state_index, state in enumerate(self.states):
            for i,so in enumerate(state.state_objs):
                if so is not None:
                    so.extract((state_index, i))


class CompositeEnvelope:
    """
    Composite Envelope is a pointer to a container, which includes the state
    Multiple Composite enveopes can point to the same containers.
    """
    _containers: Dict[Union['str', uuid.UUID], CompositeEnvelopeContainer] = {}
    _instances: Dict[Union['str', uuid.UUID], List['CompositeEnvelope']] = {}

    def __init__(self, *states: List[Union['CompositeEnvelope', 'Envelope']]):
        from photon_weave.state.envelope import Envelope
        self.uid = uuid.uuid4()
        # Check if there are composite envelopes in the argument list
        composite_envelopes:List[CompositeEnvelope] = [e for e in states if isinstance(e, CompositeEnvelope)]
        envelopes:List[Envelope] = [e for e in states if isinstance(e, Envelope)]
        for e in envelopes:
            if (e.composite_envelope is not None and
                e.composite_envelope not in composite_envelopes):
                assert isinstance(e.composite_envelope, CompositeEnvelope), "e.composite_envelope should be CompositeEnvelope type"
                composite_envelopes.append(e.composite_envelope)

        ce_container = None
        for ce in composite_envelopes:
            assert isinstance(ce, CompositeEnvelope), "ce should be CompositeEnvelope type"
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
        CompositeEnvelope._containers[self.uid] = ce_container
        if not CompositeEnvelope._instances.get(self.uid):
            CompositeEnvelope._instances[self.uid] = []
        CompositeEnvelope._instances[self.uid].append(self)
        self.update_composite_envelope_pointers()

    def __repr__(self) -> str:
        return f"CompositeEnvelope(uid={self.uid}, envelopes={[e.uid for e in self.envelopes]}, state_objects={[s.uid for s in self.state_objs]})"

    @property
    def envelopes(self) -> List['Envelope']:
        return CompositeEnvelope._containers[self.uid].envelopes

    @property
    def state_objs(self) -> List['BaseState']:
        state_objs = []
        for e in self.envelopes:
            state_objs.extend([e.fock, e.polarization])
        return state_objs

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

    def combine(self, *state_objs: 'BaseState') -> None:
        """
        Combines given states into a product state.

        Parameters
        ----------
        state_objs: BaseState
           Accepts many state_objs
        """
        from photon_weave.state.polarization import Polarization
        from photon_weave.state.fock import Fock

        # Check for the types
        for so in state_objs:
            assert isinstance(so, Fock) or isinstance(so, Polarization), f"got {type(so)}, expected Polarization or Fock type"

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
        existing_product_states = list(set(existing_product_states))
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
            elif isinstance(obj.index, int) and hasattr(obj, 'envelope'):
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
            if hasattr(so, 'envelope') and so.envelope is not None and so.index is not None and not isinstance(so.index, tuple):
                if minimum_expansion_level is ExpansionLevel.Vector:
                    state_vector_or_matrix = jnp.kron(
                        state_vector_or_matrix, so.envelope.composite_vector
                    )
                    so.envelope.composite_vector = None
                else:
                    state_vector_or_matrix = jnp.kron(
                        state_vector_or_matrix, so.envelope.composite_matrix
                    )
                    so.envelope.composite_matrix = None
                indices = [None, None]
                indices[so.envelope.fock.index] = so.envelope.fock
                indices[so.envelope.polarization.index] = so.envelope.polarization
                assert all(x is not None for x in indices), "Indices must not be None at this point"
                state_order.extend(cast(List['BaseState'],indices))
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
            container = self._containers[self.uid],
            state = state_vector_or_matrix,
            state_objs = state_order
        )

        CompositeEnvelope._containers[self.uid].states.append(ps)

        """
        Remove empty states
        """
        self.container.remove_empty_product_states()
        self.container.update_all_indices()

    def reorder(self, *ordered_states: 'BaseState') -> None:
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
            if all (s in ps.state_objs for s in ordered_states):
                states_are_combined
        if not states_are_combined:
            self.combine(*ordered_states)

        # Get the correct product state:
        ps = [p for p in self.states if all(so in p.state_objs for so in ordered_states)][0]

        # Create order
        new_order = [s for s in ps.state_objs]
        for i, ordered_state in enumerate(ordered_states):
            if new_order.index(ordered_state) != i:
                tmp = new_order[i]
                old_idx = new_order.index(ordered_state)
                new_order[i] = ordered_state
                new_order[old_idx] = tmp
        ps.reorder(*new_order)

    def measure(self, *states: 'BaseState') -> Dict['BaseState', int]:
        """
        Projective Measurement
        Given list of states will be measured, measurement is destructive
        TODO: Measurement may not be destructive for all states (qubit?)
              later on

        Parameters
        ----------
        *states: BaseState
            Dictionary of measurement outcomes, with space as key and int as result

        Returns
        -------
        Dict['BaseState', int]
            Measurement outcome for all measured states, (also implicitly measured states)
        """
        product_states = [p for p in self.states if any(so in p.state_objs for so in states)]
        outcomes: Dict['BaseState', int]
        outcomes = {}
        for ps in product_states:
            ps_states = [so for so in states if so in ps.state_objs]
            out = ps.measure(*ps_states)
            for key, item in out.items():
                outcomes[key] = item
        self._containers[self.uid].update_all_indices()
        self._containers[self.uid].remove_empty_product_states()
        return outcomes

    def measure_POVM(self, operators:List[Union[np.ndarray, jnp.ndarray]],
                     *states: 'BaseState',
                     destructive: bool = True,
                     ) -> Tuple[int, Dict['BaseState', int]]:
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
                    f"At least on Kraus operator has incorrect dimensions: {op.shape}, expected({dim},{dim})"
                )

        # Get product states
        product_states = [p for p in self.states if any(so in p.state_objs for so in states)]
        ps = None
        if len(product_states)>1:
            all_states = [s for s in states]
            for p in product_states:
                all_states.extend([s for s in p.state_objs])
            self.combine(*all_states)
            ps = [p for p in self.states if any(so in p.state_objs for so in states)][0]
        else:
            ps = product_states[0]

        # Make sure the order of the states in tensoring is correct
        self.reorder(*states)

        outcome = ps.measure_POVM(operators, *states, destructive=destructive)
        return outcome

    def apply_kraus(self, operators:List[Union[np.ndarray, jnp.ndarray]],
                    *states:'BaseState',
                    identity_check:bool=True) -> None:
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

        # Check if dimensions match
        dim = jnp.prod(jnp.array([s.dimensions for s in states]))
        for op in operators:
            if op.shape != (dim, dim):
                raise ValueError(f"At least on Kraus operator has incorrect dimensions: {op.shape}, expected({dim},{dim})")

        # Check the identity sum 
        if identity_check:
            if not kraus_identity_check(operators):
                raise ValueError("Kraus operators do not sum to the identity")


        # Get product states
        product_states = [p for p in self.states if any(so in p.state_objs for so in states)]
        ps = None
        if len(product_states)>1:
            all_states = [s for s in states]
            for p in product_states:
                all_states.extend([s for s in p.state_objs])
            self.combine(*all_states)
            ps = [p for p in self.states if any(so in p.state_objs for so in states)][0]
        else:
            ps = product_states[0]

        # Make sure the order of the states in tensoring is correct
        self.reorder(*states)

        ps.apply_kraus(operators, *states)

    def trace_out(self, *states: Union['BaseState']) -> jnp.ndarray:
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
        product_states = [p for p in self.states if any(so in p.state_objs for so in states)]
        assert len(product_states) > 0, "No product state found"
        ps: ProductState
        if len(product_states)>1:
            all_states = [s for s in states]
            for p in product_states:
                all_states.extend([s for s in p.state_objs])
            self.combine(*all_states)
            product_states = [p for p in self.states if any(so in p.state_objs for so in states)]
            assert len(product_states) > 0, "Only one product state should exist at this point"
        ps = product_states[0]

        self.reorder(*states)

        return ps.trace_out(*states)
