"""
Envelope
"""

# ruff: noqa: F401

from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Union

import numpy as np
from scipy.integrate import quad
import jax.numpy as jnp
import jax

from photon_weave.photon_weave import Config
from photon_weave.constants import C0, gaussian
from photon_weave.operation.generic_operation import GenericOperation
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.exceptions import (
    EnvelopeAlreadyMeasuredException,
    EnvelopeAssignedException,
    MissingTemporalProfileArgumentException
)

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
    def __init__(
        self,
        wavelength: float = 1550,
        fock: Optional["Fock"] = None,
        polarization: Optional["Polarization"] = None,
        temporal_profile: TemporalProfileInstance = TemporalProfile.Gaussian.with_params(
            mu=0,
            sigma=42.45 * 10 ** (-15),  # 100 fs pulse
        ),
    ):
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
        self.composite_envelope = None
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
        raise NotImplementedError("Extract method is not implemented for the envelope")

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
        if self.composite_vector is None and self.composite_matrix is None:
            self.fock.expand()
            self.polarization.expand()

    def separate(self):
        raise NotImplementedError("Sperate method is not implemented for the envelope class")

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
            outcome = (int(fock_outcome), int(polarization_outcome))
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
            outcome = (int(fock_outcome), int(polarization_outcome))
        elif isinstance(self.fock.index, (list, tuple)) and len(self.fock.index) == 2:
            outcome = self.composite_envelope.measure()
        else:
            outcome = (
                self.fock.measure(non_destructive, partial=True),
                self.polarization.measure(non_destructive))
        if not non_destructive:
            self._set_measured(remove_composite)
        return outcome

    def measure_POVM():
        raise NotImplemented()

    def apply_kraus():
        raise NotImplemented()

    def _set_measured(self, remove_composite=True):
        if self.composite_envelope is not None and remove_composite:
            self.composite_envelope.envelopes.remove(self)
            self.composite_envelope = None
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


