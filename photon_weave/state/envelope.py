"""
Envelope
"""

# ruff: noqa: F401

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import numpy as np
from scipy.integrate import quad

from photon_weave.constants import C0, gaussian
from photon_weave.operation.generic_operation import GenericOperation


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

    def __repr__(self):
        if self.measured:
            return "Envelope already Measured"
        if self.composite_matrix is None and self.composite_vector is None:
            if (
                self.fock.expansion_level == 0
                and self.polarization.expansion_level == 0
            ):
                return f"{repr(self.fock)} ⊗ {repr(self.polarization)}"
            else:
                return f"{repr(self.fock)}\n   ⊗\n {repr(self.polarization)}"
        elif self.composite_vector is not None:
            formatted_vector = "\n".join(
                [
                    f"{complex_num.real:.2f} {'+' if complex_num.imag >= 0 else '-'} {abs(complex_num.imag):.2f}j"
                    for complex_num in self.composite_vector.flatten()
                ]
            )
            return f"{formatted_vector}"
        elif self.composite_matrix is not None:
            formatted_matrix = "\n".join(
                [
                    "\t".join(
                        [
                            f"({num.real:.2f} {'+' if num.imag >= 0 else '-'} {abs(num.imag):.2f}j)"
                            for num in row
                        ]
                    )
                    for row in self.composite_matrix
                ]
            )
            return f"{formatted_matrix}"

    def combine(self):
        """
        Combines the fock and polarization into one matrix
        """
        if self.fock.expansion_level == 0:
            self.fock.expand()
        if self.polarization.expansion_level == 0:
            self.polarization.expand()

        while self.fock.expansion_level < self.polarization.expansion_level:
            self.fock.expand()

        while self.fock.expansion_level > self.polarization.expansion_level:
            self.polarization.expand()

        if self.fock.expansion_level == 1 and self.polarization.expansion_level == 1:
            self.composite_vector = np.kron(
                self.fock.state_vector, self.polarization.state_vector
            )
            self.fock.extract(0)
            self.polarization.extract(1)

        if self.fock.expansion_level == 2 and self.polarization.expansion_level == 2:
            self.composite_matrix = np.kron(
                self.fock.density_matrix, self.polarization.density_matrix
            )
            self.fock.extract(0)
            self.polarization.extract(1)

    def extract(self, state):
        pass

    @property
    def expansion_level(self):
        if self.composite_vector is not None:
            return 1
        elif self.composite_matrix is not None:
            return 2
        else:
            return -1

    def separate(self):
        pass

    def apply_operation(self, operation: GenericOperation):
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

    def measure(self, non_destructive=False, remove_composite=True):
        """
        Measures the number of particles in the space
        """
        if self.measured:
            raise EnvelopeAlreadyMeasuredException()
        outcome = None
        if self.composite_vector is not None:
            dim = [0, 0]
            dim[self.fock.index] = int(self.fock.dimensions)
            dim[self.polarization.index] = 2
            matrix_form = self.composite_vector.reshape(dim[0], dim[1])
            probabilities = np.sum(
                np.abs(matrix_form) ** 2, axis=self.polarization.index
            )
            assert np.isclose(
                np.sum(probabilities), 1.0
            ), "Probabilities do not sum to 1."
            axis = np.arange(dim[self.fock.index])
            outcome = np.random.choice(axis, p=probabilities)
        elif self.composite_matrix is not None:
            dim = [0, 0]
            dim[self.fock.index] = int(self.fock.dimensions)
            dim[self.polarization.index] = 2
            tf = self.composite_matrix.reshape(dim[0], dim[1], dim[0], dim[1])
            if self.fock.index == 0:
                tf = np.trace(tf, axis1=1, axis2=3)
            else:
                tf = np.trace(tf, axis1=0, axis2=2)
            probabilities = np.abs(np.diagonal(tf))
            axis = np.arange(dim[self.fock.index])
            outcome = np.random.choice(axis, p=probabilities)
        elif isinstance(self.fock.index, (list, tuple)) and len(self.fock.index) == 2:
            outcome = self.composite_envelope.measure()
        else:
            outcome = self.fock.measure(non_destructive)
        if not non_destructive:
            self._set_measured(remove_composite)
        return outcome

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


class EnvelopeAssignedException(Exception):
    pass


class EnvelopeAlreadyMeasuredException(Exception):
    pass


class MissingTemporalProfileArgumentException(Exception):
    pass
