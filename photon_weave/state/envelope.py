"""
Envelope
"""

# ruff: noqa: F401

from __future__ import annotations

import itertools
import logging
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from photon_weave._math.ops import (
    kraus_identity_check,
    num_quanta_matrix,
    num_quanta_vector,
)
from photon_weave.constants import C0, gaussian
from photon_weave.photon_weave import Config
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization

if TYPE_CHECKING:
    from photon_weave.operation import Operation
    from photon_weave.state.composite_envelope import CompositeEnvelope
    from photon_weave.state.polarization import PolarizationLabel

    from .base_state import BaseState

logger = logging.getLogger()


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
        logger.info("Creating Envelope with uid %s", self.uid)
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
        pol_dims = self.fock.dimensions
        return fock_dims + pol_dims

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
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            formatted_vector: Union[str, List[str]]
            formatted_vector = ""
            for row in self.state:
                formatted_row = "⎢ "
                for num in row:
                    # Format real part
                    formatted_row += f"{num.real:+.2f} "
                    if num.imag >= 0:
                        formatted_row += "+ "
                    else:
                        formatted_row += "- "

                    formatted_row += f"{abs(num.imag):.2f}j "

                formatted_row = formatted_row.strip() + " ⎥\n"
                formatted_vector += formatted_row
            formatted_vector = formatted_vector.strip().split("\n")
            formatted_vector[0] = "⎡ " + formatted_vector[0][2:-1] + "⎤"
            formatted_vector[-1] = "⎣ " + formatted_vector[-1][2:-1] + "⎦"
            formatted_vector = "\n".join(formatted_vector)

            return f"{formatted_vector}"
        elif self.state is not None and self.expansion_level == ExpansionLevel.Matrix:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, self.dimensions)
            formatted_matrix: Union[str, List[str]]
            formatted_matrix = ""

            for row in self.state:
                formatted_row = "⎢ "
                for num in row:
                    formatted_row += f"{num.real:+.2f} "
                    if num.imag >= 0:
                        formatted_row += "+ "
                    else:
                        formatted_row += "- "

                    # Format the imaginary part and add "j"
                    formatted_row += f"{abs(num.imag):.2f}j   "

                formatted_row = formatted_row.strip() + " ⎥\n"
                formatted_matrix += formatted_row

            # Add top and bottom brackets
            formatted_matrix = formatted_matrix.strip().split("\n")
            formatted_matrix[0] = "⎡" + formatted_matrix[0][1:-1] + "⎤"
            formatted_matrix[-1] = "⎣" + formatted_matrix[-1][1:-1] + "⎦"
            formatted_matrix = "\n".join(formatted_matrix)

            return f"{formatted_matrix}"
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
            self.state = jnp.kron(self.fock.state, self.polarization.state)
            self.expansion_level = ExpansionLevel.Vector
            self.fock.extract(0)
            self.polarization.extract(1)

        if (
            self.fock.expansion_level == ExpansionLevel.Matrix
            and self.polarization.expansion_level == ExpansionLevel.Matrix
        ):
            assert isinstance(self.fock.state, jnp.ndarray)
            assert isinstance(self.polarization.state, jnp.ndarray)
            self.state = jnp.kron(self.fock.state, self.polarization.state)
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

    def expand(self) -> None:
        """
        Expands the state.
        If state is in the Fock and Polarization instances
        it expands those
        """
        if self.state is None:
            self.fock.expand()
            self.polarization.expand()
            return
        if self.composite_envelope is not None:
            self.composite_envelope.expand(self.fock, self.polarization)
        if self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            self.state = jnp.dot(self.state, jnp.conj(self.state.T))
            self.expansion_level = ExpansionLevel.Matrix

    def measure(
        self,
        *states: Optional["BaseState"],
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

        outcomes = {}
        reshape_shape = []
        if self.state is None:
            for s in [self.polarization, self.fock]:
                out = s.measure()
                for k, v in out.items():
                    outcomes[k] = v
        else:
            assert isinstance(self.fock.index, int)
            assert isinstance(self.polarization.index, int)

            reshape_shape = [-1, -1]
            reshape_shape[self.fock.index] = self.fock.dimensions
            reshape_shape[self.polarization.index] = self.polarization.dimensions

            C = Config()

            if self.expansion_level == ExpansionLevel.Vector:
                assert isinstance(self.state, jnp.ndarray)
                assert self.state.shape == (self.dimensions, 1)
                reshape_shape.append(1)
                ps = self.state.reshape(reshape_shape)

                # 1. Measure Fock Part
                if (
                    (separate_measurement and self.fock in states)
                    or len(states) == 0
                    or len(states) == 2
                ):
                    probabilities = (
                        jnp.abs(jnp.sum(ps, axis=self.polarization.index)).flatten()
                        ** 2
                    )
                    key = C.random_key
                    choice = int(
                        jax.random.choice(
                            key, a=jnp.arange(len(probabilities)), p=probabilities
                        )
                    )
                    outcomes[self.fock] = choice

                    # Construct post measurement state
                    post_measurement = jnp.take(ps, choice, self.polarization.index)
                    ps = jnp.take(ps, choice, axis=self.fock.index)

                    einsum = "ij,kj->ikj"
                    if self.fock.index == 0:
                        ps = jnp.einsum(einsum, post_measurement, ps)
                    elif self.fock.index == 1:
                        ps = jnp.einsum(einsum, ps, post_measurement)

                if (
                    (separate_measurement and self.polarization in states)
                    or len(states) == 0
                    or len(states) == 2
                ):
                    probabilities = (
                        jnp.abs(jnp.sum(ps, axis=self.fock.index)).flatten() ** 2
                    )
                    key = C.random_key
                    choice = int(
                        jax.random.choice(
                            key, a=jnp.arange(len(probabilities)), p=probabilities
                        )
                    )
                    outcomes[self.polarization] = choice

                    # Construct post measurement state
                    post_measurement = jnp.take(ps, choice, self.polarization.index)
                    ps = jnp.take(ps, choice, axis=self.polarization.index)
                    einsum = "ij,kj->ikj"
                    if self.fock.index == 0:
                        ps = jnp.einsum(einsum, ps, post_measurement)
                    else:
                        ps = jnp.einsum(einsum, post_measurement, ps)

            if self.expansion_level == ExpansionLevel.Matrix:
                assert isinstance(self.state, jnp.ndarray)
                assert self.state.shape == (self.dimensions, self.dimensions)
                reshape_shape = [*reshape_shape, *reshape_shape]
                transpose_pattern = [0, 2, 1, 3]
                ps = self.state.reshape(reshape_shape).transpose(transpose_pattern)

                # 1. Measure Fock Part
                if (
                    (separate_measurement and self.fock in states)
                    or len(states) == 0
                    or len(states) == 2
                ):
                    if self.fock.index == 0:
                        subspace = jnp.einsum("bcaa->bc", ps)
                    else:
                        subspace = jnp.einsum("aabc->bc", ps)
                    probabilities = jnp.diag(subspace).real
                    probabilities /= jnp.sum(probabilities)
                    key = C.random_key
                    choice = int(
                        jax.random.choice(
                            key, a=jnp.arange(len(probabilities)), p=probabilities
                        )
                    )
                    outcomes[self.fock] = choice

                    # Reconstruct post measurement state
                    indices: List[Union[slice, int]] = [slice(None)] * len(ps.shape)
                    indices[self.fock.index] = outcomes[self.fock]
                    indices[self.fock.index + 1] = outcomes[self.fock]
                    ps = ps[tuple(indices)]

                    post_measurement = jnp.zeros(
                        (self.fock.dimensions, self.fock.dimensions)
                    )
                    post_measurement = post_measurement.at[choice, choice].set(1)
                    if self.fock.index == 0:
                        ps = jnp.einsum("ab,cd->abcd", post_measurement, ps)
                    else:
                        ps = jnp.einsum("ab,cd->abcd", ps, post_measurement)

                # 2. Measure Polarization Part
                if (
                    (separate_measurement and self.polarization in states)
                    or len(states) == 0
                    or len(states) == 2
                ):
                    if self.polarization.index == 1:
                        subspace = jnp.einsum("aabc->bc", ps)
                    else:
                        subspace = jnp.einsum("bcaa->bc", ps)
                    probabilities = jnp.diag(subspace).real
                    probabilities /= jnp.sum(probabilities)
                    key = C.random_key
                    choice = int(
                        jax.random.choice(
                            key, a=jnp.arange(len(probabilities)), p=probabilities
                        )
                    )
                    outcomes[self.polarization] = choice

                    # Reconstruct post measurement state
                    indices = [slice(None)] * len(ps.shape)
                    indices[self.polarization.index] = outcomes[self.polarization]
                    indices[self.polarization.index + 1] = outcomes[self.polarization]
                    ps = ps[tuple(indices)]

                    post_measurement = jnp.zeros(
                        (self.polarization.dimensions, self.polarization.dimensions)
                    )
                    post_measurement = post_measurement.at[choice, choice].set(1)

                    if self.polarization.index == 0:
                        ps = jnp.einsum("ab,cd->abcd", post_measurement, ps)
                    else:
                        ps = jnp.einsum("ab,cd->abcd", ps, post_measurement)

            # Handle post measurement processes
            ps = self.state.reshape(reshape_shape)
            if self.expansion_level == ExpansionLevel.Vector:
                if separate_measurement and len(states) == 1:
                    if self.fock not in states:
                        self.fock.state = jnp.take(
                            ps, outcomes[self.polarization], self.polarization.index
                        )
                        self.fock.expansion_level = ExpansionLevel.Vector
                        self.fock.index = None
                        if destructive:
                            self.polarization._set_measured()
                        else:
                            self.polarization.state = jnp.zeros((2, 1))
                            self.polarization.state.at[
                                1, outcomes[self.polarization]
                            ].set(1)
                            self.polarization.index = None
                    if self.polarization not in states:
                        self.polarization.state = jnp.take(
                            ps, outcomes[self.fock], self.fock.index
                        )
                        self.polarization.expansion_level = ExpansionLevel.Vector
                        self.polarization.index = None
                        if destructive:
                            self.fock._set_measured()
                        else:
                            self.fock.state = outcomes[self.fock]
                            self.fock.expansion_level = ExpansionLevel.Label
                            self.fock.index = None
                else:
                    if self.fock.index == 0:
                        self.fock.state = jnp.einsum("ijk->ik", ps)
                    else:
                        self.fock.state = jnp.einsum("ijk->jk", ps)
                    self.fock.expansion_level = ExpansionLevel.Vector
                    self.fock.index = None

                    if self.polarization.index == 0:
                        self.polarization.state = jnp.einsum("ijk->ik", ps)
                    else:
                        self.polarization.state = jnp.einsum("ijk->jk", ps)
                    self.polarization.expansion_level = ExpansionLevel.Vector
                    self.polarization.index = None
                    if destructive:
                        self._set_measured()
                        self.polarization._set_measured()
                        self.fock._set_measured()
            if self.expansion_level == ExpansionLevel.Matrix:
                if separate_measurement and len(states) == 1:
                    if self.fock not in states:
                        if self.fock.index == 0:
                            self.fock.state = jnp.einsum("abcb->ac", ps)
                        elif self.fock.index == 1:
                            self.fock.state = jnp.einsum("abac->bc", ps)
                        self.fock.expansion_level = ExpansionLevel.Matrix
                        self.fock.index = None
                        if destructive:
                            self.polarization._set_measured()
                    if self.polarization not in states:
                        if self.polarization.index == 0:
                            self.polarization.state = jnp.einsum("abcb->ac", ps)
                        elif self.polarization.index == 1:
                            self.polarization.state = jnp.einsum("abac->bc", ps)
                        self.polarization.expansion_level = ExpansionLevel.Matrix
                        self.polarization.index = None
                        if destructive:
                            self.fock._set_measured()
                else:
                    if self.fock.index == 0:
                        self.fock.state = jnp.einsum("ikjk->ij", ps)
                    else:
                        self.fock.state = jnp.einsum("kikj->ij", ps)
                    self.fock.expansion_level = ExpansionLevel.Matrix
                    self.fock.index = None
                    if self.polarization.index == 0:
                        self.polarization.state = jnp.einsum("ikjk->ij", ps)
                    else:
                        self.polarization.state = jnp.einsum("kikj->ij", ps)
                    self.polarization.expansion_level = ExpansionLevel.Matrix
                    self.polarization.index = None
                    if destructive:
                        self._set_measured()
                        self.fock._set_measured()
                        self.polarization._set_measured()
            self.polarization.contract()
            self.fock.contract()

        if destructive:
            self._set_measured()
        return outcomes

    def measure_POVM(
        self,
        operators: List[Union[np.ndarray, jnp.ndarray]],
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
        operators: List[Union[np.ndarray, jnp.ndarray]]
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

        self.reorder(*states)
        C = Config()

        if len(states) == 2 and self.state is None:
            self.combine()

        reshape_shape = [-1, -1]
        assert isinstance(self.fock.index, int) and isinstance(
            self.polarization.index, int
        )
        reshape_shape[self.fock.index] = self.fock.dimensions
        reshape_shape[self.polarization.index] = self.polarization.dimensions

        assert isinstance(self.state, jnp.ndarray)
        ps = self.state.reshape([*reshape_shape, *reshape_shape]).transpose(
            [0, 2, 1, 3]
        )

        # Handle POVM measurement when both spaces are measured
        if len(states) == 2:
            # Check if the dimensions match
            for op in operators:
                assert op.shape == (self.dimensions, self.dimensions)

            # Produce einsum str
            einsum = "eacf,abcd,gbhd->egfh"
            # Compute probabilities
            probabilities = []
            for op in operators:
                op = op.reshape([*reshape_shape, *reshape_shape]).transpose(
                    [0, 2, 1, 3]
                )
                prob_state = (
                    jnp.einsum(einsum, op, ps, jnp.conj(op))
                    .transpose([0, 2, 1, 3])
                    .reshape(self.dimensions, self.dimensions)
                )
                probabilities.append(jnp.trace(prob_state).real)

            probs = jnp.array(probabilities) / jnp.sum(jnp.array(probabilities))
            key = C.random_key

            choice = int(
                jax.random.choice(key, a=jnp.arange(len(operators)), p=jnp.array(probs))
            )

            # Constructing post measurement state
            op = (
                operators[choice]
                .reshape([*reshape_shape, *reshape_shape])
                .transpose([0, 2, 1, 3])
            )
            self.state = (
                jnp.einsum(einsum, op, ps, np.conj(op))
                .transpose([0, 2, 1, 3])
                .reshape((self.dimensions, self.dimensions))
            )
            self.state = self.state / jnp.trace(self.state)
            if destructive:
                self._set_measured()
                self.fock._set_measured()
                self.polarization._set_measured()
            return (choice, {})
        elif len(states) == 1:
            # Check the dimensions of the operators
            for op in operators:
                assert op.shape == (states[0].dimensions, states[0].dimensions)

            einsum = "ea,abcd,fb->efcd"
            einsum_trace = "abcc->ab"

            # Compute probabilities
            probabilities = []
            for op in operators:
                prob_state = jnp.einsum(einsum, op, ps, jnp.conj(op))
                subspace = jnp.einsum(einsum_trace, prob_state)
                probabilities.append(jnp.trace(subspace).real)
            probs = jnp.array(probabilities) / jnp.sum(jnp.array(probabilities))
            key = C.random_key
            choice = int(jax.random.choice(key, a=jnp.arange(len(operators)), p=probs))
            # Constructing post measurement state
            op = operators[choice]
            self.state = (
                jnp.einsum(einsum, op, ps, jnp.conj(op))
                .transpose([0, 2, 1, 3])
                .reshape(self.dimensions, self.dimensions)
            )
            self.state = self.state / jnp.trace(self.state)

            if destructive:
                other_state = (
                    self.fock if states[0] is self.polarization else self.polarization
                )
                ps = self.state.reshape([*reshape_shape, *reshape_shape]).transpose(
                    [0, 2, 1, 3]
                )
                assert isinstance(other_state, Fock) or isinstance(
                    other_state, Polarization
                )
                other_state.state = jnp.einsum("aabc->bc", ps)
                other_state.index = None
                other_state.expansion_level = ExpansionLevel.Matrix
                other_state.contract()
                states[0]._set_measured()
                self._set_measured()
            return (choice, {})
        # Should not come to this
        return (-1, {})  # pragma: no cover

    def apply_kraus(
        self, operators: List[Union[np.ndarray, jnp.ndarray]], *states: "BaseState"
    ) -> None:
        """
        Apply Kraus operator to the envelope.

        Parameters
        ----------
        operators: List[Union[np.ndarray, jnp.ndarray]]
            List of Kraus operators
        states:
            List of states in the same order as the tensoring of operators
        """
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

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

        # Reorder
        self.reorder(*states)

        # Kraus operators are only applied to the density matrices
        assert isinstance(self.expansion_level, ExpansionLevel)
        while self.expansion_level < ExpansionLevel.Matrix:
            self.expand()

        dim = int(jnp.prod(jnp.array([s.dimensions for s in states])))
        for op in operators:
            if op.shape != (dim, dim):
                raise ValueError("Kraus operator has incorrect dimension")

        if not kraus_identity_check(operators):
            raise ValueError(
                "Kraus operators do not sum to the identity sum K^dagg K != I"
            )

        if len(states) == 2:
            # Apply the operator fully
            assert isinstance(self.state, jnp.ndarray)
            resulting_state = jnp.zeros_like(self.state)
            for op in operators:
                resulting_state += jnp.einsum(
                    "ab,bc,dc->ad", op, self.state, np.conj(op)
                )
            self.state = resulting_state

        if len(states) == 1:
            assert isinstance(self.fock.index, int)
            assert isinstance(self.polarization.index, int)
            assert isinstance(self.state, jnp.ndarray)
            reshape_shape = [-1, -1]
            reshape_shape[self.fock.index] = self.fock.dimensions
            reshape_shape[self.polarization.index] = self.polarization.dimensions

            ps = self.state.reshape([*reshape_shape, *reshape_shape]).transpose(
                [0, 2, 1, 3]
            )
            resulting_state = jnp.zeros_like(ps)
            for op in operators:
                resulting_state += jnp.einsum("ea,abcd,fb->efcd", op, ps, np.conj(op))

            self.state = resulting_state.transpose([0, 2, 1, 3]).reshape(
                self.dimensions, self.dimensions
            )

        self.contract()

    def reorder(self, *states: "BaseState") -> None:
        """
        Changes the order of states in the product state

        Parameters
        ----------
        *states: BaseState
            new order of states
        """
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

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
            logger.info("States not combined noting to do", self.uid)
            return
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
            logger.info("States already in correct order", self.uid)
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
        assert self.state.shape == (self.dimensions, self.dimensions)
        state_squared = jnp.matmul(self.state, self.state)
        state_trace = jnp.trace(state_squared)
        if jnp.abs(state_trace - 1) < tol:
            # The state is pure
            eigenvalues, eigenvectors = jnp.linalg.eigh(self.state)
            # close_to_one = jnp.isclose(eigenvalues, 1.0, atol=tol)
            pure_state_index = jnp.argmax(jnp.abs(eigenvalues - 1.0) < tol)
            assert pure_state_index is not None, "pure_state_index should not be None"
            self.state = eigenvectors[:, pure_state_index].reshape(-1, 1)
            # Normalizing the phase
            assert self.state is not None, "self.state should not be None"
            phase = jnp.exp(-1j * jnp.angle(self.state[0]))
            self.state = self.state * phase
            self.expansion_level = ExpansionLevel.Vector

    def _set_measured(self, remove_composite: bool = True) -> None:
        if self.composite_envelope is not None and remove_composite:
            self.composite_envelope.envelopes.remove(self)
            self.composite_envelope_id = None
        self.measured = True
        self.state = None

    def trace_out(
        self, *states: "BaseState"
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

        self.reorder(*states)

        assert isinstance(self.fock.index, int)
        assert isinstance(self.polarization.index, int)

        reshape_shape = [-1, -1]
        reshape_shape[self.fock.index] = self.fock.dimensions
        reshape_shape[self.polarization.index] = self.polarization.dimensions
        state_order: List[Optional[Any]] = [None, None]
        state_order[self.fock.index] = self.fock
        state_order[self.polarization.index] = self.polarization

        if self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            reshape_shape.append(1)
            ps = self.state.reshape(reshape_shape)

            # Construct Einsum string
            c1 = itertools.count(start=0)
            einsum_list_list: List[List[int]] = [[], []]
            einsum_to = next(c1)

            for s in state_order:
                if s not in states:
                    c = einsum_to
                else:
                    c = next(c1)
                einsum_list_list[0].append(c)
                if s in states:
                    einsum_list_list[1].append(c)
            c = next(c1)
            einsum_list_list[0].append(c)
            einsum_list_list[1].append(c)
            einsum_list_str = [
                "".join([chr(97 + x) for x in s]) for s in einsum_list_list
            ]
            einsum = f"{einsum_list_str[0]}->{einsum_list_str[1]}"
            ps = jnp.einsum(einsum, ps)

            dim = int(jnp.prod(jnp.array([s.dimensions for s in states])))
            return ps.reshape(dim, 1)

        if self.expansion_level == ExpansionLevel.Matrix:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, self.dimensions)

            ps = self.state.reshape([*reshape_shape, *reshape_shape]).transpose(
                [0, 2, 1, 3]
            )

            # Construct einsum str
            c1 = itertools.count(start=0)
            einsum_list_list = [[], []]
            einsum_to = next(c1)
            for s in state_order:
                for i in range(2):
                    if s not in states:
                        c = einsum_to
                    else:
                        c = next(c1)
                    einsum_list_list[0].append(c)
                    if s in states:
                        einsum_list_list[1].append(c)
            einsum_list_str = [
                "".join([chr(97 + x) for x in s]) for s in einsum_list_list
            ]
            einsum = f"{einsum_list_str[0]}->{einsum_list_str[1]}"

            ps = jnp.einsum(einsum, ps)
            dim = int(jnp.prod(jnp.array([s.dimensions for s in states])))
            if len(states) == 2:
                ps = ps.transpose([0, 2, 1, 3])
            return ps.reshape((dim, dim))
        # Should not come to this
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

    def resize_fock(self, new_dimensions: int) -> bool:
        """
        Adjusts the dimension of the fock in the envelope.

        Parameters
        ----------
        new_dimensions: int
            New dimensions to resize to

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
                    # Cannot hrink because amplitues exist beyond new_dimensions
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

        from photon_weave.operation.fock_operation import FockOperationType
        from photon_weave.operation.polarization_operation import (
            PolarizationOperationType,
        )
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

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

        if self.state is None:
            states[0].apply_operation(operation)
            return

        self.reorder(*states)

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

        reshape_shape = [-1, -1]
        assert isinstance(self.fock.index, int)
        assert isinstance(self.fock.dimensions, int)
        assert isinstance(self.polarization.index, int)
        assert isinstance(self.polarization.dimensions, int)
        reshape_shape[self.fock.index] = self.fock.dimensions
        reshape_shape[self.polarization.index] = self.polarization.dimensions

        if self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            reshape_shape.append(1)

            ps = self.state.reshape(reshape_shape)

            # state is reordered, so the state operated on is in the first state
            ps = jnp.einsum("ij,jkl->ikl", operation.operator, ps)
            if not jnp.any(jnp.abs(ps) > 0):
                raise ValueError(
                    "The state is entirely composed of zeros, is |0⟩ attempted "
                    "to be annihilated?"
                )
            if operation.renormalize:
                ps = ps / jnp.linalg.norm(ps)
            self.state = ps.reshape((-1, 1))
            return
        if self.expansion_level == ExpansionLevel.Matrix:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, self.dimensions)
            ps = self.state.reshape([*reshape_shape, *reshape_shape]).transpose(
                [0, 2, 1, 3]
            )

            ps = jnp.einsum(
                "ij,jklm,nk->inlm", operation.operator, ps, jnp.conj(operation.operator)
            )

            ps = ps.transpose([0, 2, 1, 3])
            ps = ps.reshape(self.dimensions, self.dimensions)
            if not jnp.any(jnp.abs(ps) > 0):
                raise ValueError(
                    "The state is entirely composed of zeros, "
                    "is |0⟩ attempted to be annihilated?"
                )
            if operation.renormalize:
                ps = ps / jnp.linalg.norm(ps)
            self.state = ps.reshape((self.dimensions, self.dimensions))

            C = Config()
            if C.contractions:
                self.contract()
