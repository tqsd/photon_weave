"""
Envelope
"""

# ruff: noqa: F401

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Union, Tuple, List, TYPE_CHECKING
import logging
import uuid

import numpy as np
from scipy.integrate import quad
import jax.numpy as jnp
import jax

from photon_weave._math.ops import kraus_identity_check, apply_kraus
from photon_weave.photon_weave import Config
from photon_weave.constants import C0, gaussian
from photon_weave.operation.generic_operation import GenericOperation
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.exceptions import (
    EnvelopeAlreadyMeasuredException,
    EnvelopeAssignedException,
    MissingTemporalProfileArgumentException
)
if TYPE_CHECKING:
    from .base_state import BaseState

logger = logging.getLogger()

class TemporalProfile(Enum):
    Gaussian = (gaussian, {"mu": 0, "sigma": 1, "omega": None})

    def __init__(self, func, params):
        self.func = func
        self.params = params

    def with_params(self, **kwargs):
        params = self.params.copy()
        params.update(kwargs)
        return TemporalProfileInstance(self.func, params)


class TemporalProfileInstance:
    def __init__(self, func, params):
        self.func = func
        self.params = params

    def get_function(self, t_a, omega_a):
        params = self.params.copy()
        params.update({"t_a": t_a, "omega": omega_a})

        return lambda t: self.func(t, **params)


class Envelope:

    __slots__ = (
        "uid", "state","_expansion_level","_composite_envelope_id",
        "measured", "wavelength", "temporal_profile", "__dict__"
    )
    def __init__(
        self,
        wavelength: float = 1550,
        fock: "Fock" = None,
        polarization: "Polarization" = None,
        temporal_profile: TemporalProfileInstance = TemporalProfile.Gaussian.with_params(
            mu=0,
            sigma=42.45 * 10 ** (-15),  # 100 fs pulse
        ),
    ):
        from photon_weave.state.fock import Fock
        from photon_weave.state.polarization import Polarization

        self.uid: uuid.UUID = uuid.uuid4()
        self.composite_envelope_id = None
        logger.info("Creating Envelope with uid %s", self.uid)
        if fock is None:
            self.fock = Fock(envelope=self)
        else:
            self.fock = fock
        if polarization is None:
            self.polarization = Polarization(envelope=self)
        else:
            self.polarization = polarization
            polarization.envelope = self
        self._expansion_level: Optional(ExpansionLevel) = None
        self.composite_envelope_id = None
        self.state = None
        self.measured = False
        self.wavelength = wavelength
        self.temporal_profile = temporal_profile

    @property
    def expansion_level(self) -> Optional(ExpansionLevel):
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
            flattened_vector = self.state.flatten()
            formatted_vector: Union[str, List[str]]
            formatted_vector = "\n".join(
                [
                    f"⎢ {''.join([f'{num.real:.2f} {"+" if num.imag >= 0 else "-"} {abs(num.imag):.2f}j' for num in row])} ⎥"
                    for row in self.state
                ]
            )
            formatted_vector = formatted_vector.split("\n")
            formatted_vector[0] = "⎡" + formatted_vector[0][1:-1] + "⎤"
            formatted_vector[-1] = "⎣" + formatted_vector[-1][1:-1] + "⎦"
            formatted_vector = "\n".join(formatted_vector)
            return f"{formatted_vector}"
        elif self.state is not None and self.expansion_level == ExpansionLevel.Matrix:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, self.dimensions)
            formatted_matrix: Union[str,List[str]]
            formatted_matrix = "\n".join(
                [
                    f"⎢ {'   '.join([f'{num.real:.2f} {"+" if num.imag >= 0 else "-"} {abs(num.imag):.2f}j' for num in row])} ⎥"
                    for row in self.state
                ]
            )

            # Add top and bottom brackets
            formatted_matrix = formatted_matrix.split("\n")
            formatted_matrix[0] = "⎡" + formatted_matrix[0][1:-1] + "⎤"
            formatted_matrix[-1] = "⎣" + formatted_matrix[-1][1:-1] + "⎦"
            formatted_matrix = "\n".join(formatted_matrix)

            return f"{formatted_matrix}"

    def combine(self) -> None:
        """
        Combines the fock and polarization into one vector or matrix and
        stores it under self.composite_vector or self.composite_matrix appropriately
        """
        if self.fock.expansion_level == ExpansionLevel.Label:
            self.fock.expand()
        if self.polarization.expansion_level == ExpansionLevel.Label:
            self.polarization.expand()

        while self.fock.expansion_level < self.polarization.expansion_level:
            self.fock.expand()

        while self.polarization.expansion_level < self.fock.expansion_level:
            self.polarization.expand()

        if (self.fock.expansion_level == ExpansionLevel.Vector and
            self.polarization.expansion_level == ExpansionLevel.Vector):
            self.state = jnp.kron(
                self.fock.state, self.polarization.state
            )
            self.expansion_level = ExpansionLevel.Vector
            self.fock.extract(0)
            self.polarization.extract(1)

        if (self.fock.expansion_level == ExpansionLevel.Matrix and
            self.polarization.expansion_level == ExpansionLevel.Matrix):
            self.state= jnp.kron(
                self.fock.state, self.polarization.state
            )
            self.fock.extract(0)
            self.polarization.extract(1)
            self.expansion_level = ExpansionLevel.Matrix


    @property
    def composite_envelope(self) -> Union['CompositeEnvelope', None]:
        """
        Property, which return appropriate composite envelope
        """
        from photon_weave.state.composite_envelope import CompositeEnvelope
        if self.composite_envelope_id is not None:
            ce = CompositeEnvelope._instances[self.composite_envelope_id][0]
            assert ce is not None, "Composite Envelope should exist"
            return ce
        return None

    def set_composite_envelope_id(self, uid: uuid.UUID):
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
        if self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            self.state = jnp.dot(self.state, jnp.conj(self.state.T))
            self.expansion_level = ExpansionLevel.Matrix

    def measure(self, *states: Optional['BaseState'], separate_measurement:bool=False, destructive:bool=True) -> Dict['BaseState', int]:
        """
        Measures the envelope. If the state is measured partially, then the state are moved to their
        respective spaces. If the measurement is destructive, then the state is destroyed post measurement.

        Parameter
        ---------
        *states: Optional[BaseState]
            Optional, when measuring spaces individualy
        separate_measurement:bool
            if True given states will be measured separately and the state which is not measured will be
            preserved (False by default)
        destructive: bool 
            If False, the measurement will not destroy the state after the measurement. The state will still be
            affected by the measurement (True by default)

        Returns
        -------
        Dict[BaseState,int]
            Dictionary of outcomes, where the state is key and its outcome measurement is the value (int)
        
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

            reshape_shape = [-1,-1]
            reshape_shape[self.fock.index]=self.fock.dimensions
            reshape_shape[self.polarization.index]=self.polarization.dimensions

            C = Config()

            if self.expansion_level == ExpansionLevel.Vector:
                assert isinstance(self.state, jnp.ndarray)
                assert self.state.shape == (self.dimensions, 1)
                reshape_shape.append(1)
                ps = self.state.reshape(reshape_shape)

                # 1. Measure Fock Part
                if (separate_measurement and self.fock in states) or len(states) == 0 or len(states) == 2:
                    probabilities = jnp.abs(jnp.sum(ps, axis=self.polarization.index)).flatten()**2
                    #probabilities = jnp.abs(ps.take(self.fock.index, axis=self.fock.index).flatten())**2
                    key = C.random_key
                    out = int(jax.random.choice(
                        key,
                        a=jnp.arange(len(probabilities)),
                        p=probabilities
                    ))
                    outcomes[self.fock] = out

                    # Construct post measurement state
                    post_measurement  = jnp.take(ps, out, self.polarization.index)
                    ps = jnp.take(ps, out, axis=self.fock.index)

                    einsum = "ij,kj->ikj"
                    if self.fock.index == 0:
                        ps = jnp.einsum(einsum, post_measurement, ps)
                    elif self.fock.index == 1:
                        ps = jnp.einsum(einsum, ps, post_measurement)

                if (separate_measurement and self.polarization in states) or len(states) == 0 or len(states) == 2:
                    probabilities = jnp.abs(jnp.sum(ps, axis=self.fock.index)).flatten()**2
                    key = C.random_key
                    out = int(jax.random.choice(
                        key,
                        a=jnp.arange(len(probabilities)),
                        p=probabilities
                    ))
                    outcomes[self.polarization] = out

                    # Construct post measurement state
                    post_measurement  = jnp.take(ps, out, self.polarization.index)
                    ps = jnp.take(ps, out, axis=self.polarization.index)
                    einsum = "ij,kj->ikj"
                    if self.fock.index == 0:
                        ps = jnp.einsum(einsum, ps, post_measurement)
                    else:
                        ps = jnp.einsum(einsum, post_measurement, ps)


            if self.expansion_level == ExpansionLevel.Matrix:
                assert isinstance(self.state, jnp.ndarray)
                assert self.state.shape == (self.dimensions, self.dimensions)
                reshape_shape = [*reshape_shape, *reshape_shape]
                transpose_pattern = [0,2,1,3]
                ps = self.state.reshape(reshape_shape).transpose(transpose_pattern)

                # 1. Measure Fock Part
                if (separate_measurement and self.fock in states) or len(states) == 0 or len(states) == 2:

                    if self.fock.index == 0:
                        subspace = jnp.einsum('bcaa->bc', ps)
                    else:
                        subspace = jnp.einsum('aabc->bc', ps)
                    probabilities = jnp.diag(subspace).real
                    probabilities /= jnp.sum(probabilities)
                    key = C.random_key
                    out = int(jax.random.choice(
                        key,
                        a=jnp.arange(len(probabilities)),
                        p=probabilities
                    ))
                    outcomes[self.fock] = out

                    # Reconstruct post measurement state
                    indices: List[Union[slice, int]] = [slice(None)]*len(ps.shape)
                    indices[self.fock.index] = outcomes[self.fock]
                    indices[self.fock.index+1] = outcomes[self.fock]
                    ps = ps[tuple(indices)]

                    post_measurement = jnp.zeros((self.fock.dimensions, self.fock.dimensions))
                    post_measurement = post_measurement.at[out,out].set(1)
                    if self.fock.index == 0:
                        ps = jnp.einsum('ab,cd->abcd', post_measurement,  ps)
                    else:
                        ps = jnp.einsum('ab,cd->abcd', ps,  post_measurement)
                        


                # 2. Measure Polarization Part
                if (separate_measurement and self.polarization in states) or len(states) == 0 or len(states) == 2:
                    if self.polarization.index == 1:
                        subspace = jnp.einsum('aabc->bc', ps)
                    else:
                        subspace = jnp.einsum('bcaa->bc', ps)
                    probabilities = jnp.diag(subspace).real
                    probabilities /= jnp.sum(probabilities)
                    key = C.random_key
                    out = int(jax.random.choice(
                        key,
                        a=jnp.arange(len(probabilities)),
                        p=probabilities
                    ))
                    outcomes[self.polarization] = out

                    # Reconstruct post measurement state
                    indices: List[Union[slice, int]] = [slice(None)]*len(ps.shape)
                    indices[self.polarization.index] = outcomes[self.polarization]
                    indices[self.polarization.index+1] = outcomes[self.polarization]
                    ps = ps[tuple(indices)]

                    post_measurement = jnp.zeros((self.polarization.dimensions, self.polarization.dimensions))
                    post_measurement = post_measurement.at[out,out].set(1)

                    if self.polarization.index == 0:
                        ps = jnp.einsum('ab,cd->abcd', post_measurement,  ps)
                    else:
                        ps = jnp.einsum('ab,cd->abcd', ps,  post_measurement)

            # Handle post measurement processes
            ps = self.state.reshape(reshape_shape)
            if self.expansion_level == ExpansionLevel.Vector:
                if separate_measurement and len(states) == 1:
                    if self.fock not in states:
                        self.fock.state = jnp.take(ps, outcomes[self.polarization], self.polarization.index)
                        self.fock.expansion_level = ExpansionLevel.Vector
                        self.fock.index = None
                        if destructive:
                            self.polarization._set_measured()
                    if self.polarization not in states:
                        self.polarization.state = jnp.take(ps, outcomes[self.fock], self.fock.index)
                        self.polarization.expansion_level = ExpansionLevel.Vector
                        self.polarization.index = None
                        if destructive:
                            self.fock._set_measured()
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
            if self.expansion_level == ExpansionLevel.Matrix:
                if separate_measurement and len(states) == 1:
                    if self.fock not in states:
                        if self.fock.index == 0:
                            self.fock.state = jnp.einsum('abcb->ac', ps)
                        elif self.fock.index == 1:
                            self.fock.state = jnp.einsum('abac->bc', ps)
                        self.fock.expansion_level = ExpansionLevel.Matrix
                        self.fock.index = None
                        if destructive:
                            self.polarization._set_measured()
                    if self.polarization not in states:
                        if self.polarization.index == 0:
                            self.polarization.state = jnp.einsum(
                                'abcb->ac', ps
                            )
                        elif self.polarization.index == 1:
                            self.polarization.state = jnp.einsum(
                                'abac->bc', ps
                            )
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
            self.polarization.contract()
            self.fock.contract()

        if destructive:
            self._set_measured()
        return outcomes

    def measure_POVM(self, operators: List[Union[np.ndarray, jnp.ndarray]],
                     states: 'BaseState',
                     destructive:bool=False) -> Tuple[int, Dict['BaseState', int]]:
        """
        Positive Operation-Valued Measurement,
        POVM measurement does not destroy the quantum object by default.

        Parameters
        ----------
        operators: List[Union[np.ndarray, jnp.ndarray]]
        states:Tuple[Union[np.ndarray, jnp.ndarray],
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
        from photon_weave.state.polarization import Polarization
        from photon_weave.state.fock import Fock
        if len(states) == 2:
            if ((isinstance(states[0], Fock) and isinstance(states[1], Fock)) or
                (isinstance(states[0], Polarization) and isinstance(states[1], Polarization))):
                raise ValueError("Given states have to be unique")
        elif len(states) > 2:
            raise ValueError("Too many states given")
        for s in states:
            if s is not self.polarization and s is not self.fock:
                raise ValueError("Given states have to be members of the envelope, use env.fock and env.polarization")
        # Handle partial measurement
        if len(states) == 1 and self.composite_matrix is None and self.composite_vector is None:
            if isinstance(states[0], Fock):
                outcome = self.fock.measure_POVM(operators)
            elif isinstance(states[0], Polarization):
                outcome = self.polarization.measure_POVM(operators)
            return outcome

        if self.composite_matrix is None and self.composite_vector is None:
            self.combine()

        if self.composite_vector is not None and self.composite_matrix is None:
            self.expand()
        self.reorder(states.copy())
        C = Config()
        # Handle POVM measurement when both spaces are measured
        if len(states) == 2:
            probabilities = jnp.array([
                jnp.trace(jnp.matmul(op, self.composite_matrix)).real for op in operators
            ])
            probabilities = probabilities / jnp.sum(probabilities)
            key = C.random_key

            measurement_result = jax.random.choice(
                key,
                a = jnp.arange(len(operators)),
                p = probabilities
            )
            if destructive:
                self._set_measured()
            return int(measurement_result)
        elif len(states) == 1:
            # Perform partial measurement
            self.reorder([self.fock, self.polarization])
            if isinstance(states[0], Polarization):
                # Trace Out Fock Space
                reduced_state = jnp.trace(
                    self.composite_matrix.reshape(self.fock.dimensions,2,self.fock.dimensions,2),
                    axis1=1, axis2=3
                )
            elif isinstance(states[0], Fock):
                # Trace Out Fock Space
                reduced_state = jnp.trace(
                    self.composite_matrix.reshape(self.fock.dimensions,2,self.fock.dimensions,2),
                    axis1=0, axis2=2
                )

            probabilities = jnp.array([
                jnp.trace(jnp.matmul(op, reduced_state)).real for op in operators
            ])
            probabilities = probabilities / jnp.sum(probabilities)
            key = C.random_key
            measurement_result = jax.random.choice(
                key,
                a=jnp.arange(len(operators)),
                p=probabilities
            )
            selected_operator = operators[int(measurement_result)]
            post_measurement_state = jnp.matmul(
                selected_operator,
                jnp.matmul(reduced_state, selected_operator.T))

            if states[0] is self.polarization:
                operator1 = jnp.kron(jnp.eye(self.fock.dimensions), selected_operator)
                operator2 = jnp.kron(jnp.eye(self.fock.dimensions), jnp.transpose(jnp.conj(selected_operator)))
            elif state[0] is self.fock:
                operator1 = jnp.kron(selected_operator, jnp.eye(self.fock.dimensions))
                operator2 = jnp.kron(jnp.transpose(jnp.conj(selected_operator)), jnp.eye(self.fock.dimensions))

            self.composite_matrix = jnp.matmul(operator1, jnp.matmul(self.composite_matrix, operator2))
            self.composite_matrix /= jnp.trace(self.composite_matrix)
            
            if destructive:
                self._set_measured()




    def apply_kraus(self,
                    operators: List[Union[np.ndarray, jnp.ndarray]],
                    states: Tuple[Union["Fock", "Polarization"], Union["Polarization", "Fock"]]) -> None:
        """
        Apply Kraus operator to the envelope.

        Parameters
        ----------
        operators: List[Union[np.ndarray, jnp.ndarray]]
            List of Kraus operators
        states:
        """
        from photon_weave.state.polarization import Polarization
        from photon_weave.state.fock import Fock
        if len(states) == 2:
            if ((isinstance(states[0], Fock) and isinstance(states[1], Fock)) or
                (isinstance(states[0], Polarization) and isinstance(states[1], Polarization))):
                raise ValueError("Given states have to be unique")
        elif len(states) > 2:
            raise ValueError("Too many states given")
        for s in states:
            if s is not self.polarization and s is not self.fock:
                raise ValueError("Given states have to be members of the envelope, use env.fock and env.polarization")

        # Handle partial application 
        if len(states) == 1 and self.composite_matrix is None and self.composite_vector is None:
            if isinstance(states[0], Fock):
                self.fock.apply_kraus(operators)
            elif isinstance(states[0], Polarization):
                self.polarization.apply_kraus(operators)
            return

        # Combine the states if Kraus operators are applied to both states
        if self.composite_vector is None and self.composite_matrix is None:
            self.combine()

        # Reorder
        self.reorder(states)

        # Kraus operators are only applied to the density matrices
        if self.composite_vector is not None and self.composite_matrix is None:
            self.expand()

        # Check operators
        dim = self.composite_matrix.shape[0]
        for op in operators:
            if op.shape != (dim, dim):
                raise ValueError(f"Kraus operator has incorrect dimension")

        if not kraus_identity_check(operators):
            raise ValueError("Kraus operators do not sum to the identity sum K^dagg K != I")

        self.composite_matrix = apply_kraus(self.composite_matrix, operators)
        self.contract()

    def reorder(self, *states: 'BaseState') -> None:
        """
        Changes the order of states in the product state

        Parameters
        ----------
        *states: BaseState
            new order of states
        """
        from photon_weave.state.polarization import Polarization
        from photon_weave.state.fock import Fock
        states = list(states)
        if len(states) == 2:
            if ((isinstance(states[0], Fock) and isinstance(states[1], Fock)) or
                (isinstance(states[0], Polarization) and isinstance(states[1], Polarization))):
                raise ValueError("Given states have to be unique")
        elif len(states) > 2:
            raise ValueError("Too many states given")

        for s in states:
            if s is not self.polarization and s is not self.fock:
                raise ValueError("Given states have to be members of the envelope, use env.fock and env.polarization")

        if (self.state is None):
            logger.info("States not combined noting to do", self.uid)
            return
        if len(states) == 1:
            if states[0] is self.fock:
                states.append(self.polarization)
            elif states[0] is self.polarization:
                states.append(self.fock)

        current_order = [None, None]
        current_order[self.fock.index] = self.fock
        current_order[self.polarization.index] = self.polarization

        if current_order[0] is states[0] and current_order[1] is states[1]:
            logger.info("States already in correct order", self.uid)
            return
            
        current_shape = [0,0]
        current_shape[self.fock.index] = self.fock.dimensions
        current_shape[self.polarization.index] = 2

        if self.expansion_level == ExpansionLevel.Vector:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, 1)
            tmp_vector = self.state.reshape((current_shape[0], current_shape[1]))
            tmp_vector = jnp.transpose(tmp_vector, (1,0))
            self.state = tmp_vector.reshape(-1,1)
            self.fock.index, self.polarization.index = self.polarization.index, self.fock.index
        elif self.expansion_level == ExpansionLevel.Matrix:
            assert isinstance(self.state, jnp.ndarray)
            assert self.state.shape == (self.dimensions, self.dimensions)
            tmp_matrix = self.state.reshape(
                current_shape[0], current_shape[1], current_shape[0], current_shape[1]
            )
            tmp_matrix = jnp.transpose(tmp_matrix, (1,0,3,2))
            self.state= tmp_matrix.reshape((current_shape[0]*current_shape[1] for i in range(2)))
            self.fock.index, self.polarization.index = self.polarization.index, self.fock.index


    def contract(self) -> None:
        """
        Attempt contracting the state, if the state is encoded in matrix form
        """
        pass
    
    def _set_measured(self, remove_composite=True):
        if self.composite_envelope is not None and remove_composite:
            self.composite_envelope.envelopes.remove(self)
            self.composite_envelope_id = None
        self.measured = True
        self.state = None

    def overlap_integral(self, other: Envelope, delay: float, n: float = 1):
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
        result, error = quad(integrand, -np.inf, np.inf)

        return result


