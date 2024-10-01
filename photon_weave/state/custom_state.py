"""
This module implements Custom State

Custom state can be used to implement various quantum system
such as qubits or quantum dots
"""

import uuid
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from photon_weave._math.ops import apply_kraus, kraus_identity_check
from photon_weave.operation import CustomStateOperationType, Operation
from photon_weave.photon_weave import Config
from photon_weave.state.base_state import BaseState
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.expansion_levels import ExpansionLevel


class CustomState(BaseState):
    def __init__(self, dimensions: int):
        self.uid = uuid.uuid4()
        # Custom state can only be separate or part of composite envelope
        self.index: Optional[Tuple[int, int]] = None
        self._dimensions_set = False
        self.dimensions = dimensions
        # Initialize in the |0> state
        self.state: Optional[Union[int, jnp.ndarray]] = 0
        self.expansion_level = ExpansionLevel.Label
        self.composite_envelope: Optional["CompositeEnvelope"] = None

    @property
    def uid(self) -> Union[str, uuid.UUID]:
        return self._uid

    @uid.setter
    def uid(self, uid: Union[uuid.UUID, str]) -> None:
        self._uid = uid

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dimensions: int) -> None:
        if self._dimensions_set:
            raise ValueError("Dimensions can only be set once")
        self._dimensions = dimensions
        self._dimensions_set = True

    @property
    def index(self) -> Union[None, int, Tuple[int, int]]:
        assert not isinstance(self._index, int)
        return self._index

    @index.setter
    def index(self, index: Union[None, int, Tuple[int, int]]) -> None:
        assert not isinstance(index, int)
        self._index = index

    def expand(self) -> None:
        """
        Expands the state from label to vector and from vector to matrix
        """
        if isinstance(self.index, tuple):
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            self.composite_envelope.expand(self)
        if self.expansion_level == ExpansionLevel.Label:
            assert isinstance(self.state, int)
            assert self.state >= 0 and self.state < self.dimensions
            index_value = self.state
            self.state = jnp.zeros((self.dimensions, 1))
            self.state = self.state.at[index_value].set(1.0)
            self.expansion_level = ExpansionLevel.Vector
        elif self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            self.state = jnp.dot(self.state, self.state.T)
            self.expansion_level = ExpansionLevel.Matrix

    def contract(
        self, final: ExpansionLevel = ExpansionLevel.Label, tol: float = 1e-6
    ) -> None:
        """
        Attempts to contract the representation to the level defined in final argument.

        Parameters
        ----------
        final: ExpansionLevel
            Expected expansion level after contraction
        tol: float
            Tolerance when comparing matrices
        """
        if (
            self.expansion_level is ExpansionLevel.Matrix
            and final < ExpansionLevel.Matrix
        ):
            # Check if the state is pure state
            assert isinstance(self.state, jnp.ndarray), "self.state should be a ndarray"
            assert self.state.shape == (
                self.dimensions,
                self.dimensions,
            ), "Dimensions do not match"
            state_squared = jnp.matmul(self.state, self.state)
            state_trace = jnp.trace(state_squared)
            if jnp.abs(state_trace - 1) < tol:
                # The state is pure
                eigenvalues, eigenvectors = jnp.linalg.eigh(self.state)
                pure_state_index = jnp.argmax(jnp.abs(eigenvalues - 1.0) < tol)
                assert (
                    pure_state_index is not None
                ), "pure_state_index should not be None"
                self.state = eigenvectors[:, pure_state_index].reshape(-1, 1)
                assert isinstance(self.state, jnp.ndarray)
                phase = jnp.exp(-1j * jnp.angle(self.state[0]))
                self.state = self.state * phase
                self.expansion_level = ExpansionLevel.Vector
        if (
            self.expansion_level is ExpansionLevel.Vector
            and final < ExpansionLevel.Vector
        ):
            assert isinstance(self.state, jnp.ndarray), "self.state should be a ndarray"
            assert self.state.shape == (self.dimensions, 1)
            ones = jnp.where(self.state == 1)[0]
            if ones.size == 1:
                self.state = int(ones[0])
                self.expansion_level = ExpansionLevel.Label

    @property
    def _measured(self) -> bool:
        return False

    @_measured.setter
    def _measured(self, measured: bool) -> None:
        """
        Configured property, without effect
        Custom state cannot be destroyed, so
        measured is without effect
        """
        pass

    def _set_measured(self) -> None:
        pass

    def extract(self, index: Union[int, Tuple[int, int]]) -> None:
        assert isinstance(index, tuple)
        assert len(index) == 2
        self.index = index
        self.state = None

    def set_index(self, minor: int = -1, major: int = -1) -> None:
        if minor == -1 and major == -1:
            self.index = None
        elif major >= 0 and minor >= 0:
            self.index = (major, minor)
        else:
            raise ValueError(
                "Either set both parameters (minor, major) or none of them"
            )

    def measure(
        self, separate_measurement: bool = False, destructive: bool = True
    ) -> Dict[BaseState, int]:
        """
        Measures the state and returns an outcome in a dict

        Parameters
        ----------
        separate_measurement: bool
            Doesn't have any effect, implemented to not defy the signature
            of the superclass
        destructive: bool
            Doesn't have any effect, implemented to not defy the signature
            of the superclass

        Returns
        -------
        Dict[BaseState, int]
            Dictionary of outcomes, in this case only one outcome
        """
        if self.index is not None:
            assert self.composite_envelope is not None
            return self.composite_envelope.measure(self)
        elif self.index is None:
            if self.expansion_level == ExpansionLevel.Label:
                assert isinstance(self.state, int)
                return {self: self.state}
            elif self.expansion_level == ExpansionLevel.Vector:
                assert isinstance(self.state, jnp.ndarray)
                assert self.state.shape == (self.dimensions, 1)
                C = Config()
                probabilities = jnp.abs(self.state.flatten()) ** 2
                probabilities = probabilities.ravel()
                assert jnp.isclose(sum(probabilities), 1)
                key = C.random_key
                out = int(
                    jax.random.choice(
                        key, a=jnp.array(len(probabilities)), p=probabilities
                    )
                )
                self.state = out
                self.expansion_level = ExpansionLevel.Label
                return {self: out}
            elif self.expansion_level == ExpansionLevel.Matrix:
                assert isinstance(self.state, jnp.ndarray)
                assert self.state.shape == (self.dimensions, self.dimensions)
                C = Config()
                probabilities = jnp.diag(self.state).real
                probabilities = probabilities / jnp.sum(probabilities)
                key = C.random_key
                out = int(
                    jax.random.choice(
                        key, a=jnp.array(len(probabilities)), p=probabilities
                    )
                )
                self.state = out
                self.expansion_level = ExpansionLevel.Label
                return {self: out}
        raise ValueError(
            "Something went wrong, this exception should not be raised"
        )  # pragma: no cover

    def measure_POVM(
        self,
        operators: List[Union[np.ndarray, jnp.ndarray]],
        destructive: bool = True,
        partial: bool = False,
    ) -> Tuple[int, Dict[BaseState, int]]:
        """
        Positive Operation-Valued Measurement

        Parameters
        ----------
        *operators: Union[np.ndarray, jnp.Array]
            List of the POVM measurement operators
        destructive: bool
            Does not have an effect on custom state, implemented only to satisfy the
            superclass method signature
        partial: bool
            Does not have an effect on custom state, implemented only to satisfy the
            superclass method signature

        Returns
        -------
        Tuple[int, Dict[BaseState, int]]
            Tuple, where first value is the outcome of this POVM measurement,
            second is empty, used when envelope parts are measured
        """
        if self.index is not None:
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            return self.composite_envelope.measure_POVM(operators, self)

        for op in operators:
            assert op.shape == (self.dimensions, self.dimensions)

        assert isinstance(self.expansion_level, ExpansionLevel)
        while self.expansion_level < ExpansionLevel.Matrix:
            self.expand()

        C = Config()

        assert isinstance(self.state, jnp.ndarray)
        probabilities = jnp.array(
            [jnp.trace(jnp.matmul(op, self.state)).real for op in operators]
        )
        probabilities = probabilities / jnp.sum(probabilities)

        key = C.random_key
        outcome = int(
            jax.random.choice(key, a=jnp.arange(len(operators)), p=probabilities)
        )
        self.state = jnp.matmul(
            operators[outcome], jnp.matmul(self.state, jnp.conj(operators[outcome].T))
        )
        self.state = self.state / jnp.trace(self.state)

        if C.contractions:
            self.contract()
        return (outcome, {})

    def apply_kraus(
        self,
        operators: List[Union[np.ndarray, jnp.ndarray]],
        identity_check: bool = True,
    ) -> None:
        """
        Apply Kraus operators to the state. The State is automatically expanded to the
        density matrix representation.

        Parameters
        ----------
        operators: List[Union[np.ndarray, jnp.Array]]
            List of the operators
        identity_check: bool
            Signal to check whether or not the operators sum up to identity, True by
            default
        """
        if self.index is not None:
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            self.composite_envelope.apply_kraus(operators, self)

        while self.expansion_level != ExpansionLevel.Matrix:
            self.expand()

        for op in operators:
            if op.shape != (self.dimensions, self.dimensions):
                raise ValueError(
                    f"Kraus operator has incorrect dimensions: {op.shape}, "
                    f"expected ({self.dimensions},{self.dimensions})"
                )

        if not kraus_identity_check(operators):
            raise ValueError("Kraus operators do not sum to the identity")

        self.state = apply_kraus(self.state, operators)
        C = Config()
        if C.contractions:
            self.contract()

    def apply_operation(self, operation: Operation) -> None:
        """
        Applies an operation to the state. If state is in some product
        state, the operator is correctly routed to the specific
        state

        Parameters
        ----------
        operation: Operation
            Operation with operation type: FockOperationType
        """
        assert isinstance(operation._operation_type, CustomStateOperationType)

        if isinstance(self.index, tuple):
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            self.composite_envelope.apply_operation(operation, self)
            return

        assert isinstance(self.expansion_level, ExpansionLevel)
        while self.expansion_level < operation.required_expansion_level:
            self.expand()

        # Consolidate the dimensions
        to = self.trace_out()
        assert isinstance(to, jnp.ndarray)
        operation.compute_dimensions(0, to)

        assert operation.operator.shape == (self.dimensions, self.dimensions)

        if self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            self.state = jnp.einsum("ij,jk->ik", operation.operator, self.state)
            if not jnp.any(jnp.abs(self.state) > 0):
                raise ValueError(
                    "The state is entirely composed of zeros,"
                    " is |0⟩ attempted to be anniilated?"
                )
            # cummulative = 0
            if operation.renormalize:
                self.state = self.state / jnp.linalg.norm(self.state)
        if self.expansion_level == ExpansionLevel.Matrix:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, self.dimensions)
            self.state = jnp.einsum(
                "ca,ab,db->cd",
                operation.operator,
                self.state,
                jnp.conj(operation.operator),
            )
            if not jnp.any(jnp.abs(self.state) > 0):
                raise ValueError(
                    "The state is entirely composed of zeros, "
                    "is |0⟩ attempted to be anniilated?"
                )
            if operation.renormalize:
                self.state = self.state / jnp.linalg.norm(self.state)

        C = Config()
        if C.contractions:
            self.contract()
