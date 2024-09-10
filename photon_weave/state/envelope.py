"""
Envelope
"""

# ruff: noqa: F401

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Union, Tuple, List
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
        "uid", "composite_vector", "composite_matrix",
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
        self.uid: uuid.UUID = uuid.uuid4()
        self.composite_envelope_id = None
        logger.info("Creating Envelope with uid %s", self.uid)
        if fock is None:
            from .fock import Fock

            self.fock = Fock(envelope=self)
        else:
            self.fock = fock

        if polarization is None:
            from .polarization import Polarization

            self.polarization = Polarization(envelope=self)
        else:
            self.polarization = polarization
            polarization.envelope = self

        self.composite_vector = None
        self.composite_matrix = None
        self.composite_envelope_id = None
        self.measured = False
        self.wavelength = wavelength
        self.temporal_profile = temporal_profile

    def __repr__(self) -> str:
        if self.measured:
            return "Envelope already measured"
        if self.composite_matrix is None and self.composite_vector is None:
            fock_repr = self.fock.__repr__().splitlines()
            pol_repr = self.polarization.__repr__().splitlines()
            # Maximum length accross fock repr
            max_length = max(len(line) for line in fock_repr)
            max_lines = max(len(fock_repr), len(pol_repr))

            fock_repr.extend([" " * max_length] * (max_lines - len(fock_repr)))
            pol_repr.extend([""] * (max_lines - len(pol_repr)))
            zipped_lines = zip(fock_repr, pol_repr)
            return "\n".join(f"{f_line} ⊗ {p_line}" for f_line, p_line in zipped_lines)

        elif self.composite_vector is not None:
            flattened_vector = self.composite_vector.flatten()
            formatted_vector: Union[str, List[str]]
            formatted_vector = "\n".join(
                [
                    f"⎢ {''.join([f'{num.real:.2f} {"+" if num.imag >= 0 else "-"} {abs(num.imag):.2f}j' for num in row])} ⎥"
                    for row in self.composite_vector
                ]
            )
            formatted_vector = formatted_vector.split("\n")
            formatted_vector[0] = "⎡" + formatted_vector[0][1:-1] + "⎤"
            formatted_vector[-1] = "⎣" + formatted_vector[-1][1:-1] + "⎦"
            formatted_vector = "\n".join(formatted_vector)
            return f"{formatted_vector}"
        elif self.composite_matrix is not None:
            formatted_matrix: Union[str,List[str]]
            formatted_matrix = "\n".join(
                [
                    f"⎢ {'   '.join([f'{num.real:.2f} {"+" if num.imag >= 0 else "-"} {abs(num.imag):.2f}j' for num in row])} ⎥"
                    for row in self.composite_matrix
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
            self.composite_vector = jnp.kron(
                self.fock.state_vector, self.polarization.state_vector
            )
            self.fock.extract(0)
            self.polarization.extract(1)

        if (self.fock.expansion_level == ExpansionLevel.Matrix and
            self.polarization.expansion_level == ExpansionLevel.Matrix):
            self.composite_matrix = jnp.kron(
                self.fock.density_matrix, self.polarization.density_matrix
            )
            self.fock.extract(0)
            self.polarization.extract(1)

    def extract(self, state) -> None:
        raise NotImplementedError("Extract method is not implemented for the envelope") # pragma: no cover

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

    @property
    def expansion_level(self) -> Union[ExpansionLevel, int]:
        if self.composite_vector is not None:
            return ExpansionLevel.Vector
        elif self.composite_matrix is not None:
            return ExpansionLevel.Matrix
        else:
            return -1

    def expand(self) -> None:
        """
        Expands the state.
        If state is in the Fock and Polarization instances
        it expands those
        """
        if self.composite_vector is not None:
            self.composite_matrix= jnp.outer(
                self.composite_vector.flatten(), jnp.conj(self.composite_vector.flatten())
            )
            self.composite_vector = None
            self.fock.expansion_level = ExpansionLevel.Matrix
            self.polarization.expansion_level = ExpansionLevel.Matrix
        if self.composite_vector is None and self.composite_matrix is None:
            self.fock.expand()
            self.polarization.expand()

    def separate(self):
        raise NotImplementedError("Sperate method is not implemented for the envelope class") # pragma: no cover

    def apply_operation(self, operation: GenericOperation) -> None:
        """
        Applies operation to the system.
        Depending on the operation type, the operation will be applied to the polarization
        or to the Fock space.

        Parameters
        ----------
        operation: GenericOperation
            Operation that will be applied
        """
        from photon_weave.operation.fock_operation import (
            FockOperation,
            FockOperationType,
        )
        from photon_weave.operation.polarization_operations import (
            PolarizationOperation,
            PolarizationOperationType,
        )

        if isinstance(operation, FockOperation):
            if self.composite_vector is None and self.composite_matrix is None:
                self.fock.apply_operation(operation)
            else:
                fock_index = self.fock.index
                polarization_index = self.polarization.index
                operation.compute_operator(self.fock.dimensions)
                operators = [1, 1]
                operators[fock_index] = operation.operator
                polarization_identity = PolarizationOperation(
                    operation=PolarizationOperationType.I
                )
                polarization_identity.compute_operator()
                operators[polarization_index] = polarization_identity.operator
                operator = np.kron(*operators)
                if self.composite_vector is not None:
                    self.composite_vector = operator @ self.composite_vector
                    if operation.renormalize:
                        nf = np.linalg.norm(self.composite_vector)
                        self.composite_vector = self.composite_vector / nf
                if self.composite_matrix is not None:
                    self.composite_matrix = operator @ self.composite_matrix
                    op_dagger = operator.conj().T
                    self.composite_matrix = self.composite_matrix @ op_dagger
                    if operation.renormalize:
                        nf = np.linalg.norm(self.composite_matrix)
                        self.composite_matrix = self.composite_matrix / nf
        if isinstance(operation, PolarizationOperation):
            if self.composite_vector is None and self.composite_matrix is None:
                self.polarization.apply_operation(operation)
            else:
                fock_index = self.fock.index
                polarization_index = self.polarization.index
                operators = [1, 1]
                fock_identity = FockOperation(operation=FockOperationType.Identity)
                fock_identity.compute_operator(self.fock.dimensions)
                operators[polarization_index] = operation.operator
                operators[fock_index] = fock_identity.operator
                operator = np.kron(*operators)
                if self.composite_vector is not None:
                    self.composite_vector = operator @ self.composite_vector
                if self.composite_matrix is not None:
                    self.composite_matrix = operator @ self.composite_matrix
                    op_dagger = operator.conj().T
                    self.composite_matrix = self.composite_matrix @ op_dagger

    def measure(self, non_destructive=False, remove_composite=True) -> Tuple[int, int]:
        """
        Measures the number of particles in the space

        """
        if self.measured:
            raise EnvelopeAlreadyMeasuredException()
        outcome = [-1,-1]
        outcomes = {}
        if self.composite_vector is not None:
            C = Config()
            dim = [0, 0]
            dim[self.fock.index] = int(self.fock.dimensions)
            dim[self.polarization.index] = 2
            matrix_form = self.composite_vector.reshape(dim[0], dim[1])
            fock_probabilities = jnp.sum(
                jnp.abs(matrix_form)**2, axis = self.polarization.index)
            assert jnp.isclose(jnp.sum(fock_probabilities), 1.0)
            fock_axis = jnp.arange(dim[self.fock.index])
            key = C.random_key
            fock_outcome = jax.random.choice(key, a=fock_axis, p=fock_probabilities)

            polarization_probabilities = jnp.abs(matrix_form[fock_outcome, :]) **2
            polarization_probabilities /= jnp.sum(polarization_probabilities)

            polarization_axis = jnp.arange(dim[self.polarization.index])
            key = C.random_key
            polarization_outcome = jax.random.choice(key, a=polarization_axis, p=polarization_probabilities)
            outcomes[self.fock] = int(fock_outcome)
            outcomes[self.polarization] =  int(polarization_outcome)
            self.fock._set_measured()
            self.polarization._set_measured()
        elif self.composite_matrix is not None:
            C = Config()
            dim = [0, 0]
            dim[self.fock.index] = int(self.fock.dimensions)
            dim[self.polarization.index] = 2
            matrix_form = self.composite_matrix.reshape(
                dim[self.fock.index], dim[self.polarization.index],
                dim[self.fock.index], dim[self.polarization.index]
            ).transpose(0,2,1,3)
            fock_probabilities = jnp.einsum('ijkk->ij', matrix_form)
            fock_probabilities = jnp.sum(jnp.real(matrix_form), axis=(2, 3))
            fock_probabilities = jnp.sum(fock_probabilities, axis=1)
            fock_probabilities = fock_probabilities.real
            assert jnp.isclose(jnp.sum(fock_probabilities), 1)
            fock_axis = jnp.arange(dim[self.fock.index])
            key = C.random_key
            fock_outcome = jax.random.choice(
                key, a=fock_axis, p=fock_probabilities
            )

            polarization_probabilities = jnp.diag(matrix_form[fock_outcome, fock_outcome, :, :])
            polarization_probabilities /= jnp.sum(polarization_probabilities)
            polarization_probabilities = polarization_probabilities.real
            polarization_axis = jnp.arange(dim[self.polarization.index])
            key = C.random_key
            polarization_outcome = jax.random.choice(
                key,
                a=polarization_axis,
                p=polarization_probabilities
            )
            outcomes[self.fock] =int(fock_outcome)
            outcomes[self.polarization] =  int(polarization_outcome)
            self.fock._set_measured()
            self.polarization._set_measured()
        for s in [self.fock, self.polarization]:
            if isinstance(s.index, (list, tuple)) and len(s.index)==2:
                if not s.measured:
                    out = self.composite_envelope.measure(s)
                    for key, value in out.items():
                        if value is not None:
                            outcomes[key] = value
            else:
                if not s.measured:
                    out = s.measure()
                    for key, value in out.items():
                        outcomes[key] = value
        if not non_destructive:
            self._set_measured(remove_composite)
        return outcomes

    def measure_POVM(self, operators: List[Union[np.ndarray, jnp.ndarray]],
                     states:Tuple[Union[np.ndarray, jnp.ndarray],
                                  Optional[Union[np.ndarray, jnp.ndarray]]],
                     destructive:bool=False) -> int:
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

    def reorder(self, states: Tuple[Union["Fock", "Polarization"], Union["Fock", "Polarization"]]) -> None:
        """
        Changes the order of states in the product state
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

        if (self.composite_matrix is None) and (self.composite_vector is None):
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

        if not self.composite_vector is None:
            tmp_vector = self.composite_vector.reshape((current_shape[0], current_shape[1]))
            tmp_vector = jnp.transpose(tmp_vector, (1,0))
            self.composite_vector = tmp_vector.reshape(-1,1)
            self.fock.index, self.polarization.index = self.polarization.index, self.fock.index
        elif not self.composite_matrix is None:
            tmp_matrix = self.composite_matrix.reshape(
                current_shape[0], current_shape[1], current_shape[0], current_shape[1]
            )
            tmp_matrix = jnp.transpose(tmp_matrix, (1,0,3,2))
            self.composite_matrix = tmp_matrix.reshape((current_shape[0]*current_shape[1] for i in range(2)))
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
        self.composite_vector = None
        self.composite_matrix = None
        self.fock._set_measured()
        self.polarization._set_measured()

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


