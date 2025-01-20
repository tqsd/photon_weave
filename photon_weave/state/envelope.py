"""
Envelope
"""

# ruff: noqa: F401

from __future__ import annotations

import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.integrate import quad

from photon_weave._math.ops import (
    kraus_identity_check,
    num_quanta_matrix,
    num_quanta_vector,
)
from photon_weave.constants import C0, gaussian
from photon_weave.operation import (
    FockOperationType,
    Operation,
    PolarizationOperationType,
)
from photon_weave.photon_weave import Config
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization

from .utils.measurements import measure_matrix, measure_POVM_matrix, measure_vector
from .utils.operations import (
    apply_kraus_matrix,
    apply_kraus_vector,
    apply_operation_matrix,
    apply_operation_vector,
)
from .utils.representation import representation_matrix, representation_vector
from .utils.state_transform import state_contract, state_expand
from .utils.trace_out import trace_out_matrix, trace_out_vector

if TYPE_CHECKING:
    # from photon_weave.operation import Operation
    from photon_weave.state.composite_envelope import CompositeEnvelope
    from photon_weave.state.polarization import PolarizationLabel

    from .base_state import BaseState

jitted_kron = jit(jnp.kron)


class TemporalProfile(Enum):
    Gaussian = (gaussian, {"mu": 0, "sigma": 1, "omega": None})

    def __init__(self, func: Callable, params: Any):
        self.func = func
        self.parameters = params

    def with_params(self, **kwargs: Any) -> TemporalProfileInstance:
        params = self.parameters.copy()
        params.update(kwargs)
        return TemporalProfileInstance(self.func, params)


class TemporalProfileInstance:
    def __init__(self, func: Callable, params: Dict[Any, Any]) -> None:
        self.func = func
        self.params = params

    def get_function(self, t_a: float, omega_a: float) -> Callable:
        params = self.params.copy()
        params.update({"t_a": t_a, "omega": omega_a})

        return lambda t: self.func(t, **params)


default_temporal_profile = TemporalProfile.Gaussian.with_params(
    mu=0,
    sigma=42.45 * 10 ** (-15),  # 100 fs pulse
)


class Envelope:
    __slots__ = (
        "uid",
        "state",
        "_expansion_level",
        "_composite_envelope_id",
        "measured",
        "wavelength",
        "temporal_profile",
        "__dict__",
    )

    def __init__(
        self,
        wavelength: float = 1550,
        fock: Optional["Fock"] = None,
        polarization: Optional["Polarization"] = None,
        temporal_profile: TemporalProfileInstance = default_temporal_profile,
    ):
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        self.uid: uuid.UUID = uuid.uuid4()
        self.fock = Fock() if fock is None else fock
        self.fock.envelope = self
        self.polarization = Polarization() if polarization is None else polarization
        self.polarization.envelope = self
        self._expansion_level: Optional[ExpansionLevel] = None
        self.composite_envelope_id: Optional[uuid.UUID] = None
        self.state: Optional[jnp.ndarray] = None
        self.measured = False
        self.wavelength = wavelength
        self.temporal_profile = temporal_profile

    @property
    def expansion_level(self) -> Optional[ExpansionLevel]:
        if self._expansion_level is not None:
            return self._expansion_level
        if self.fock.expansion_level == self.polarization.expansion_level:
            return self.fock.expansion_level
        return None

    @expansion_level.setter
    def expansion_level(self, expansion_level: ExpansionLevel) -> None:
        self._expansion_level = expansion_level
        self.fock.expansion_level = expansion_level
        self.polarization.expansion_level = expansion_level

    @property
    def dimensions(self) -> int:
        fock_dims = self.fock.dimensions
        pol_dims = self.polarization.dimensions
        return fock_dims * pol_dims

    def __repr__(self) -> str:
        if self.measured:
            return "Envelope already measured"
        if self.state is None:
            fock_repr = self.fock.__repr__().splitlines()
            pol_repr = self.polarization.__repr__().splitlines()
            # Maximum length accross fock repr
            max_length = max(len(line) for line in fock_repr)
            max_lines = max(len(fock_repr), len(pol_repr))

            fock_repr.extend([" " * max_length] * (max_lines - len(fock_repr)))
            pol_repr.extend([""] * (max_lines - len(pol_repr)))
            zipped_lines = zip(fock_repr, pol_repr)
            return "\n".join(f"{f_line} ⊗ {p_line}" for f_line, p_line in zipped_lines)
        elif self.state is not None and self.expansion_level == ExpansionLevel.Vector:
            return representation_vector(self.state)
        elif self.state is not None and self.expansion_level == ExpansionLevel.Matrix:
            return representation_matrix(self.state)
        return str(self.uid)  # pragme: no cover

    def combine(self) -> None:
        """
        Combines the fock and polarization into one vector or matrix and
        stores it under self.composite_vector or self.composite_matrix appropriately
        """
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        for s in [self.fock, self.polarization]:
            if s.measured:
                raise ValueError(
                    "Parts of this envelope have already been destructively measured,"
                    " cannot combine"
                )

        assert isinstance(self.fock.expansion_level, ExpansionLevel)
        assert isinstance(self.polarization.expansion_level, ExpansionLevel)
        if self.fock.expansion_level == ExpansionLevel.Label:
            self.fock.expand()
        if self.polarization.expansion_level == ExpansionLevel.Label:
            self.polarization.expand()

        while self.fock.expansion_level < self.polarization.expansion_level:
            self.fock.expand()

        while self.polarization.expansion_level < self.fock.expansion_level:
            self.polarization.expand()

        if (
            self.fock.expansion_level == ExpansionLevel.Vector
            and self.polarization.expansion_level == ExpansionLevel.Vector
        ):
            assert isinstance(self.fock, Fock)
            assert hasattr(self.fock, "state") and isinstance(
                self.fock.state, jnp.ndarray
            )
            assert isinstance(self.polarization, Polarization)
            assert isinstance(self.polarization.state, jnp.ndarray)

            self.state = jitted_kron(self.fock.state, self.polarization.state)
            self.expansion_level = ExpansionLevel.Vector
            self.fock.extract(0)
            self.polarization.extract(1)

        if (
            self.fock.expansion_level == ExpansionLevel.Matrix
            and self.polarization.expansion_level == ExpansionLevel.Matrix
        ):
            assert isinstance(self.fock.state, jnp.ndarray)
            assert isinstance(self.polarization.state, jnp.ndarray)
            # fock_broadcast = self.fock.state[:,:,None,None]
            # polarization_broadcast = self.polarization.state[None,None,:,:]
            # self.state = fock_broadcast * polarization_broadcast
            self.state = jitted_kron(self.fock.state, self.polarization.state)
            self.fock.extract(0)
            self.polarization.extract(1)
            self.expansion_level = ExpansionLevel.Matrix

    @property
    def composite_envelope(self) -> Union["CompositeEnvelope", None]:
        """
        Property, which return appropriate composite envelope
        """
        from photon_weave.state.composite_envelope import CompositeEnvelope

        if self.composite_envelope_id is not None:
            ce = CompositeEnvelope._instances[self.composite_envelope_id][0]
            assert ce is not None, "Composite Envelope should exist"
            return ce
        return None

    def set_composite_envelope_id(self, uid: uuid.UUID) -> None:
        self.composite_envelope_id = uid

    def expand(self, *args: Any, **kwargs: Any) -> None:
        """
        Expands the state.
        If state is in the Fock and Polarization instances
        it expands those

        Parameters
        ----------
        *args:  Tuple
            Unused argument for compatibility, with routing functionality
        **kwargs: Dict
            Unused argument for compatibility, with routing functionality

        """
        if self.state is None:
            self.fock.expand()
            self.polarization.expand()
        else:
            assert isinstance(self.expansion_level, ExpansionLevel)
            self.state, self.expansion_level = state_expand(
                self.state, self.expansion_level, self.dimensions
            )

    def measure(
        self,
        *states: "BaseState",
        separate_measurement: bool = False,
        destructive: bool = True,
    ) -> Dict["BaseState", int]:
        """
        Measures the envelope. If the state is measured partially, then the state are
        moved to their respective spaces. If the measurement is destructive, then the
        state is destroyed post measurement.

        Parameter
        ---------
        *states: Optional[BaseState]
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
        if self.measured:
            raise ValueError("Envelope has already been destroyed")

        # Check if given states are part of this envelope
        for s in states:
            assert s in [self.fock, self.polarization]
        if len(states) == 0:
            states = (self.fock, self.polarization)

        outcomes = {}
        if self.state is None:
            for s in [self.polarization, self.fock]:
                out = s.measure()
                for k, v in out.items():
                    outcomes[k] = v
        else:
            self.reorder(self.fock, self.polarization)
            if not separate_measurement and len(states) == 1:
                states = (self.fock, self.polarization)
            if len(states) == 2:
                separate_measurement = True

            assert all(isinstance(s.index, int) for s in states)

            match self.expansion_level:
                case ExpansionLevel.Vector:
                    outcomes, self.state = measure_vector(
                        [self.fock, self.polarization], list(states), self.state
                    )
                case ExpansionLevel.Matrix:
                    outcomes, self.state = measure_matrix(
                        [self.fock, self.polarization], list(states), self.state
                    )

            # Post Measurement process
            C = Config()
            for s in [self.fock, self.polarization]:
                if separate_measurement and s not in states:
                    s.state = self.state
                    s.index = None
                    s.expansion_level = self.expansion_level
                    # Try to contract the state twice
                    if C.contractions:
                        s.contract()
                        s.contract()
                else:
                    s.state = jnp.zeros((s.dimensions, 1))
                    s.state = s.state.at[outcomes[s], 0].set(1)
                    s.expansion_level = ExpansionLevel.Vector
                    s.index = None
                    if destructive:
                        s._set_measured()
                    else:
                        while (
                            s.expansion_level > ExpansionLevel.Label
                        ) and C.contractions:
                            s.contract()

        if destructive:
            self._set_measured()

        return outcomes

    def measure_POVM(
        self,
        operators: List[jnp.ndarray],
        *states: "BaseState",
        destructive: bool = True,
    ) -> Tuple[int, Dict["BaseState", int]]:
        """
        Positive Operation-Valued Measurement
        The measured state is destroyed (discarded) by default. If not, then
        it is affected by the measurement, but not discarded. If measuring
        two spaces this method automatically combines the two spaces if
        not already combined.

        Parameters
        ----------
        operators: List[jnp.ndarray]
            List of the POVM operators
        *states:Tuple[Union[np.ndarray, jnp.ndarray],
                     Optional[Union[np.ndarray, jnp.ndarray]]
            States on which the POVM measurement should be carried out,
            Order of the states must resemble order of the operator tensoring
        destructive: bool
            If True then after the measurement the density matrix is discarded

        Returns
        -------
        int
            The index of the measurement corresponding to the outcome
        """
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        if self.measured:
            raise ValueError("This envelope was already measured")

        # Check the validity of the given states
        if len(states) == 2:
            if (isinstance(states[0], Fock) and isinstance(states[1], Fock)) or (
                isinstance(states[0], Polarization)
                and isinstance(states[1], Polarization)
            ):
                raise ValueError("Given states have to be unique")
        elif len(states) > 2:
            raise ValueError("Too many states given")
        for s in states:
            if s is not self.polarization and s is not self.fock:
                raise ValueError(
                    "Given states have to be members of the envelope, "
                    "use env.fock and env.polarization"
                )

        # Handle partial uncombined measurement
        if len(states) == 1 and self.state is None:
            if isinstance(states[0], Fock):
                outcome = self.fock.measure_POVM(operators, destructive=destructive)
            elif isinstance(states[0], Polarization):
                outcome = self.polarization.measure_POVM(
                    operators, destructive=destructive
                )
            return outcome

        # Handle the measurement if the state is in composite envelope product state
        if self.composite_envelope_id is not None:
            assert self.composite_envelope is not None
            return self.composite_envelope.measure_POVM(operators, *states)

        # Expand to matrix state if not alreay in it
        assert isinstance(self.expansion_level, ExpansionLevel)
        while self.expansion_level < ExpansionLevel.Matrix:
            self.expand()

        if len(states) == 2 and self.state is None:
            self.combine()

        assert isinstance(self.fock.index, int) and isinstance(
            self.polarization.index, int
        )
        assert isinstance(self.state, jnp.ndarray)
        self.reorder(self.fock, self.polarization)
        outcome, self.state = measure_POVM_matrix(
            [self.fock, self.polarization], list(states), operators, self.state
        )
        if len(states) == 2:
            if destructive:
                self._set_measured()
                self.fock._set_measured()
                self.polarization._set_measured()
            return (outcome, {})
        else:
            if destructive:
                other_state = (
                    self.fock if states[0] is self.polarization else self.polarization
                )
                other_state.state = trace_out_matrix(
                    [self.fock, self.polarization], [other_state], self.state
                )
                other_state.index = None
                other_state.expansion_level = ExpansionLevel.Matrix
                other_state.contract()
                states[0]._set_measured()
                self._set_measured()
        return (outcome, {})

    def apply_kraus(
        self,
        operators: List[jnp.ndarray],
        *states: BaseState,
        validity_check: bool = True,
    ) -> None:
        """
        Apply Kraus operator to the envelope.

        Parameters
        ----------
        operators: List[Union[np.ndarray, jnp.ndarray]]
            List of Kraus operators
        states:
            List of states in the same order as the tensoring of operators
        validity_check: bool
            Checks if given channel is valid, True by default
        """
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        if len(states) == 0:
            raise ValueError("At least one state must be defined")

        s_uids = [s.uid for s in states]
        if (len(states) == 2) and (
            (self.fock.uid not in s_uids) or (self.polarization.uid not in s_uids)
        ):
            raise ValueError("Both given states must belong to the Envelope")
        elif len(states) > 2:
            raise ValueError("Too many states given")

        # If any of the states is in bigger product state apply the kraus there
        if self.state is None and any(isinstance(s.index, tuple) for s in states):
            assert self.composite_envelope_id is not None
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            self.composite_envelope.apply_kraus(operators, *states)
            return

        # Handle partial application
        if len(states) == 1 and self.state is None:
            if isinstance(states[0], Fock):
                self.fock.apply_kraus(operators)
            elif isinstance(states[0], Polarization):
                self.polarization.apply_kraus(operators)
            return

        # Combine the states if Kraus operators are applied to both states
        if self.state is None:
            self.combine()
        assert isinstance(self.state, jnp.ndarray)

        if validity_check:
            if not kraus_identity_check(operators):
                raise ValueError(
                    "Kraus operators do not sum to the identity sum K^dagg K != I"
                )

        # Reorder
        self.reorder(*states)

        state_objs: List[Optional[BaseState]] = [None, None]
        state_objs[self.fock.index] = self.fock  # type: ignore
        state_objs[self.polarization.index] = self.polarization  # type: ignore

        # Kraus operators are only applied to the density matrices
        match self.expansion_level:
            case ExpansionLevel.Vector:
                self.state = apply_kraus_vector(
                    state_objs, states, self.state, operators  # type: ignore
                )
            case ExpansionLevel.Matrix:
                self.state = apply_kraus_matrix(
                    state_objs, states, self.state, operators  # type: ignore
                )

        C = Config()
        if C.contractions:
            self.contract()

    def reorder(self, *states: "BaseState") -> None:
        """
        Changes the order of states in the product state

        Parameters
        ----------
        *states: BaseState
            new order of states
        """
        states_list = list(states)
        if len(states_list) == 2:
            if (
                isinstance(states_list[0], Fock) and isinstance(states_list[1], Fock)
            ) or (
                isinstance(states_list[0], Polarization)
                and isinstance(states_list[1], Polarization)
            ):
                raise ValueError("Given states have to be unique")
        elif len(states_list) > 2:
            raise ValueError("Too many states given")

        for s in states_list:
            if s not in [self.polarization, self.fock]:
                raise ValueError(
                    "Given states have to be members of the envelope, "
                    "use env.fock and env.polarization"
                )

        if self.state is None:
            return

        # Creating the new order in the product space
        if len(states_list) == 1:
            if states_list[0] is self.fock:
                states_list.append(self.polarization)
            elif states_list[0] is self.polarization:
                states_list.append(self.fock)

        assert isinstance(self.fock.index, int)
        assert isinstance(self.polarization.index, int)
        assert isinstance(self.fock.dimensions, int)
        current_order: List[Optional[Any]] = [None, None]
        current_order[self.fock.index] = self.fock
        current_order[self.polarization.index] = self.polarization

        if current_order[0] is states_list[0] and current_order[1] is states_list[1]:
            return

        current_shape = [0, 0]
        current_shape[self.fock.index] = self.fock.dimensions
        current_shape[self.polarization.index] = 2

        if self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            tmp_vector = self.state.reshape((current_shape[0], current_shape[1]))
            tmp_vector = jnp.transpose(tmp_vector, (1, 0))
            self.state = tmp_vector.reshape(-1, 1)
            self.fock.index, self.polarization.index = (
                self.polarization.index,
                self.fock.index,
            )
        elif self.expansion_level == ExpansionLevel.Matrix:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, self.dimensions)
            tmp_matrix = self.state.reshape(
                current_shape[0], current_shape[1], current_shape[0], current_shape[1]
            )
            tmp_matrix = jnp.transpose(tmp_matrix, (1, 0, 3, 2))
            self.state = tmp_matrix.reshape(
                (current_shape[0] * current_shape[1] for i in range(2))
            )
            self.fock.index, self.polarization.index = (
                self.polarization.index,
                self.fock.index,
            )

    def contract(
        self, final: ExpansionLevel = ExpansionLevel.Vector, tol: float = 1e-6
    ) -> None:
        """
        Attempts to contract the representation to the level defined in `final`argument.

        Parameters
        ----------
        final: ExpansionLevel
            Expected expansion level after contraction
        tol: float
            Tolerance when comparing matrices
        """
        # Will not attempt to contract past vector
        # final = ExpansionLevel.Vector
        assert isinstance(self.state, jnp.ndarray)
        if self.expansion_level == ExpansionLevel.Vector:
            return
        assert self.state.shape == (self.dimensions, self.dimensions)
        if self.expansion_level == ExpansionLevel.Matrix:
            self.state, self.expansion_level, success = state_contract(  # type: ignore
                self.state, self.expansion_level
            )

    def _set_measured(self, remove_composite: bool = True) -> None:
        if self.composite_envelope is not None and remove_composite:
            self.composite_envelope.envelopes.remove(self)
            self.composite_envelope_id = None
        self.measured = True
        self.state = None

    def trace_out(
        self, *states: BaseState
    ) -> Union[jnp.ndarray, int, "PolarizationLabel"]:
        """
        Traces out the system, returning only the given states,
        if states are not in the same product space, then they
        are combined.

        Parameters
        ----------
        *states: BaseState
            The given states will be returned in the given
            order (tensoring order), with the rest traced out
        """
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization, PolarizationLabel

        if self.composite_envelope is not None:
            assert isinstance(self.composite_envelope, CompositeEnvelope)
            return self.composite_envelope.trace_out(*states)

        if self.state is None and self.composite_envelope is None:
            if len(states) == 1:
                assert isinstance(states[0], (Polarization, Fock))
                assert isinstance(
                    states[0].state, (int, PolarizationLabel, jnp.ndarray)
                )
                return states[0].state
            elif len(states) == 2:
                self.combine()

        self.reorder(self.fock, self.polarization)

        assert isinstance(self.fock.index, int)
        assert isinstance(self.polarization.index, int)
        assert self.state is not None

        match self.expansion_level:
            case ExpansionLevel.Vector:
                return trace_out_vector(
                    [self.fock, self.polarization], states, self.state
                )
            case ExpansionLevel.Matrix:
                return trace_out_matrix(
                    [self.fock, self.polarization], states, self.state
                )
        return jnp.ndarray([[1]])  # pragma: no cover

    def overlap_integral(self, other: Envelope, delay: float, n: float = 1) -> float:
        r"""
        Given delay in [seconds] this method computes overlap of temporal
        profiles between this envelope and other envelope.

        Args:
        self (Envelope): Self
        other (Envelope): Other envelope to compute overlap with
        delay (float): Delay of the `other`after self
        Returns:
        float: overlap factor
        """
        f1 = self.temporal_profile.get_function(
            t_a=0, omega_a=(C0 / n) / self.wavelength
        )
        f2 = other.temporal_profile.get_function(
            t_a=delay, omega_a=(C0 / n) / other.wavelength
        )
        integrand = lambda x: np.conj(f1(x)) * f2(x)
        result, _ = quad(integrand, -np.inf, np.inf)

        return result

    def resize_fock(self, new_dimensions: int, state: Optional[Fock] = None) -> bool:
        """
        Adjusts the dimension of the fock in the envelope. The dimensions are adjusted
        also when the fock space is in the product state.

        Parameters
        ----------
        new_dimensions: int
            New dimensions to resize to
        state: Optional[Fock]
            Optional Parameter, not needed, because there can only be one
            Fock state in an Envelope, but needed because of the routing
            logic

        Returns
        -------
        bool
            True if resizing succeeded
        """

        if self.state is None:
            return self.fock.resize(new_dimensions)

        reshape_shape = [-1, -1]
        assert isinstance(self.fock.dimensions, int)
        assert isinstance(self.fock.index, int)
        assert isinstance(self.polarization.dimensions, int)
        assert isinstance(self.polarization.index, int)
        reshape_shape[self.fock.index] = self.fock.dimensions
        reshape_shape[self.polarization.index] = self.polarization.dimensions

        if self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            reshape_shape.append(1)
            ps = self.state.reshape(reshape_shape)
            if new_dimensions > self.fock.dimensions:
                padding = new_dimensions - self.fock.dimensions
                pad_config = [(0, 0) for _ in range(ps.ndim)]
                pad_config[self.fock.index] = (0, padding)
                ps = jnp.pad(ps, pad_config, mode="constant", constant_values=0)
                self.state = ps.reshape(-1, 1)
                self.fock.dimensions = new_dimensions
                return True
            if new_dimensions < self.fock.dimensions:
                to = self.trace_out(self.fock)
                assert isinstance(to, jnp.ndarray)
                num_quanta = num_quanta_vector(to)
                if num_quanta >= new_dimensions:
                    # Cannot shrink because amplitues exist beyond new_dimensions
                    return False
                slices = [slice(None)] * ps.ndim
                slices[self.fock.index] = slice(0, new_dimensions)
                ps = ps[tuple(slices)]
                self.state = ps.reshape(-1, 1)
                self.fock.dimensions = new_dimensions
                return True
        if self.expansion_level == ExpansionLevel.Matrix:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, self.dimensions)
            ps = self.state.reshape([*reshape_shape, *reshape_shape]).transpose(
                [0, 2, 1, 3]
            )
            if new_dimensions > self.fock.dimensions:
                padding = new_dimensions - self.fock.dimensions
                pad_config = [(0, 0) for _ in range(ps.ndim)]
                pad_config[self.fock.index * 2] = (0, padding)
                pad_config[self.fock.index * 2 + 1] = (0, padding)

                ps = jnp.pad(ps, pad_config, mode="constant", constant_values=0)
                self.fock.dimensions = new_dimensions
                ps = ps.transpose([0, 2, 1, 3])
                self.state = ps.reshape((self.dimensions, self.dimensions))
                return True
            if new_dimensions <= self.fock.dimensions:
                to = self.trace_out(self.fock)
                assert isinstance(to, jnp.ndarray)
                num_quanta = num_quanta_matrix(to)
                if num_quanta >= new_dimensions:
                    return False
                slices = [slice(None)] * ps.ndim
                slices[self.fock.index * 2] = slice(0, new_dimensions)
                slices[self.fock.index * 2 + 1] = slice(0, new_dimensions)

                ps = ps[tuple(slices)]
                ps = ps.transpose([0, 2, 1, 3])
                self.fock.dimensions = new_dimensions
                self.state = ps.reshape((self.dimensions, self.dimensions))
                return True
        return False  # pragma: no cover

    def apply_operation(
        self, operation: "Operation", *states: Union["Fock", "Polarization"]
    ) -> None:
        """
        Applies given operation to the correct state

        Parameters
        ----------
        operation: Operation
            Operation to be applied to the state
        state: Union[Fock, Polarization]
            The state to which the operator should be applied to
        """

        # Check that correct operation is applied to the correct system
        if isinstance(operation._operation_type, FockOperationType):
            if not isinstance(states[0], Fock):
                raise ValueError(
                    "Cannot apply {type(operation._operation_type)} to "
                    f"{type(states[0])}"
                )
        if isinstance(operation._operation_type, PolarizationOperationType):
            if not isinstance(states[0], Polarization):
                raise ValueError(
                    "Cannot apply {type(operation._operation_type)} to "
                    f"{type(states[0])}"
                )

        # If the system is held in a different state container,
        # then apply it there
        if self.state is None:
            states[0].apply_operation(operation)
            return

        if not jnp.any(jnp.abs(self.state) > 0):
            raise ValueError("The state is invalid." "The state only consists of 0s.")

        # Compute the operator
        if isinstance(operation._operation_type, FockOperationType) and isinstance(
            states[0], Fock
        ):
            to = self.fock.trace_out()
            assert isinstance(to, jnp.ndarray)
            operation.compute_dimensions(states[0]._num_quanta, to)
            states[0].resize(operation.dimensions[0])
        elif isinstance(
            operation._operation_type, PolarizationOperationType
        ) and isinstance(states[0], Polarization):
            # Given arguments 0,0 don't have an effect
            operation.compute_dimensions([0], jnp.array([0]))

        assert isinstance(self.fock.index, int)
        assert isinstance(self.fock.dimensions, int)
        assert isinstance(self.polarization.index, int)
        assert isinstance(self.polarization.dimensions, int)
        current_order_dict: Dict[int, BaseState] = {}
        current_order_dict[self.fock.index] = self.fock
        current_order_dict[self.polarization.index] = self.polarization
        current_order = [current_order_dict[i] for i in sorted(current_order_dict)]
        # current_order: List[Optional[BaseState]] = [None, None]
        # current_order[self.fock.index] = self.fock
        # current_order[self.polarization.index] = self.polarization
        match self.expansion_level:
            case ExpansionLevel.Vector:
                self.state = apply_operation_vector(
                    current_order, list(states), self.state, operation.operator
                )
            case ExpansionLevel.Matrix:
                self.state = apply_operation_matrix(
                    current_order, list(states), self.state, operation.operator
                )

        if not jnp.any(jnp.abs(self.state) > 0):
            raise ValueError(
                "The state is entirely composed of zeros, "
                "is |0⟩ attempted to be annihilated?"
            )
        if operation.renormalize:
            self.state = self.state / jnp.linalg.norm(self.state)
        C = Config()
        if C.contractions:
            self.contract()
