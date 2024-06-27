"""
Fock state
"""

from __future__ import annotations

import numpy as np

from photon_weave.operation.fock_operation import FockOperation, FockOperationType

from .envelope import EnvelopeAssignedException
from .expansion_levels import ExpansionLevel


class Fock:
    def __init__(self, envelope: "Envelope" = None):
        """
        Creates Fock object in a vacuum state
        """
        self.dimensions = -1
        self.index = None
        self.label = 0
        self.state_vector = None
        self.density_matrix = None
        self.envelope = envelope
        self.expansion_level = ExpansionLevel.Label
        self.measured = False

    def __repr__(self):
        if self.label is not None:
            return f"|{self.label}⟩"
        elif self.state_vector is not None:
            formatted_vector = "\n".join(
                [
                    f"{complex_num.real:.2f} {'+' if complex_num.imag >= 0 else '-'} {abs(complex_num.imag):.2f}j"
                    for complex_num in self.state_vector.flatten()
                ]
            )

            return f"{formatted_vector}"
        elif self.density_matrix is not None:
            formatted_matrix = "\n".join(
                [
                    "\t".join(
                        [
                            f"({num.real:.2f} {'+' if num.imag >= 0 else '-'} {abs(num.imag):.2f}j)"
                            for num in row
                        ]
                    )
                    for row in self.density_matrix
                ]
            )
            return f"{formatted_matrix}"
        elif self.index is not None:
            return "System is part of the Envelope"
        else:
            return "Invalid Fock object"

    def __eq__(self, other: "Optional"):
        if not isinstance(other, Fock):
            return False
        if self.label is not None and other.label is not None:
            if self.label == other.label:
                return True
            return False
        if self.state_vector is not None and other.state_vector is not None:
            if np.array_equal(self.state_vector, other.state_vector):
                return True
            return False
        if self.density_matrix is not None and other.density_matrix is not None:
            if np.array_equal(self.density_matrix, other.density_matrix):
                return True
            return False
        return False

    def assign_envelope(self, envelope: "Envelope"):
        from .envelope import Envelope

        assert isinstance(envelope, Envelope)
        if self.envelope is not None:
            raise EnvelopeAssignedException("Envelope can't be reassigned")
        self.envelope = envelope

    def expand(self):
        if self.dimensions < 0:
            self.dimensions = self.label + 3
        if self.expansion_level is ExpansionLevel.Label:
            state_vector = np.zeros(int(self.dimensions))
            state_vector[self.label] = 1
            self.state_vector = state_vector[:, np.newaxis]
            self.label = None
            self.expansion_level = ExpansionLevel.Vector
        elif self.expansion_level is ExpansionLevel.Vector:
            self.density_matrix = np.outer(
                self.state_vector.flatten(), np.conj(self.state_vector.flatten())
            )
            self.state_vector = None
            self.expansion_level = ExpansionLevel.Matrix

    @property
    def expansion_level_old(self):
        if self.label is not None:
            return 0
        elif self.state_vector is not None:
            return 1
        elif self.density_matrix is not None:
            return 2
        else:
            return self.envelope.expansion_level

    def extract(self, index: int):
        self.index = index
        self.label = None
        self.density_matrix = None
        self.state_vector = None

    def apply_operation(self, operation: FockOperation) -> None:
        match operation.operation:
            case FockOperationType.Creation:
                if self.label is not None:
                    self.label += operation.apply_count
                    return
            case FockOperationType.Annihilation:
                if self.label is not None:
                    self.label -= operation.apply_count
                    if self.label < 0:
                        self.label = 0
                    return
        min_expansion_level = operation.expansion_level_required()
        while self.expansion_level < min_expansion_level:
            self.expand()

        cutoff_required = operation.cutoff_required(self._num_quanta)
        if cutoff_required > self.dimensions:
            self.resize(cutoff_required)

        match operation.operation:
            case FockOperationType.Creation:
                if self._num_quanta + operation.apply_count + 1 > self.dimensions:
                    self.resize(self._num_quanta + operation.apply_count + 1)
        operation.compute_operator(self.dimensions)

        self._execute_apply(operation)

    @property
    def _num_quanta(self):
        """
        returns highest basis with non_zero probability
        """
        if self.state_vector is not None:
            non_zero_indices = np.nonzero(self.state_vector)[
                0
            ]  # Get indices of non-zero elements
            highest_non_zero_index_vector = non_zero_indices[-1]
            return highest_non_zero_index_vector
        if self.density_matrix is not None:
            non_zero_rows = np.any(self.density_matrix != 0, axis=1)
            non_zero_cols = np.any(self.density_matrix != 0, axis=0)

            highest_non_zero_index_row = (
                np.where(non_zero_rows)[0][-1] if np.any(non_zero_rows) else None
            )
            highest_non_zero_index_col = (
                np.where(non_zero_cols)[0][-1] if np.any(non_zero_cols) else None
            )

            # Determine the overall highest index
            highest_non_zero_index_matrix = max(
                highest_non_zero_index_row, highest_non_zero_index_col
            )
            return highest_non_zero_index_matrix

    def _execute_apply(self, operation: FockOperation):
        """
        Consider GPU
        """
        if self.state_vector is not None:
            self.state_vector = operation.operator @ self.state_vector
        if self.density_matrix is not None:
            self.density_matrix = operation.operator @ self.density_matrix
            self.density_matrix @= operation.operator.conj().T

        if operation.renormalize:
            self.normalize()

    def normalize(self):
        if self.density_matrix is not None:
            trace_rho = np.trace(self.density_matrix)
            self.density_matrix = self.density_matrix / trace_rho
        if self.state_vector is not None:
            norm_psi = np.linalg.norm(self.state_vector)
            self.state_vector = self.state_vector / norm_psi

    def resize(self, new_dimensions):
        if self.label is not None:
            self.dimensions = new_dimensions
            return
        # Pad
        if self.dimensions < new_dimensions:
            pad_size = new_dimensions - self.dimensions
            if self.state_vector is not None:
                self.state_vector = np.pad(
                    self.state_vector,
                    ((0, pad_size), (0, 0)),
                    "constant",
                    constant_values=(0,),
                )
            if self.density_matrix is not None:
                self.density_matrix = np.pad(
                    self.density_matrix,
                    ((0, pad_size), (0, pad_size)),
                    "constant",
                    constant_values=0,
                )
        # Truncate
        elif self.dimensions > new_dimensions:
            pad_size = new_dimensions - self.dimensions
            if self.state_vector is not None:
                if np.all(self.state_vector[new_dimensions:] == 0):
                    self.state_vector = self.state_vector[:new_dimensions]
            if self.density_matrix is not None:
                bottom_rows_zero = np.all(self.density_matrix[new_dimensions:, :] == 0)
                right_columns_zero = np.all(
                    self.density_matrix[:, new_dimensions:] == 0
                )
                if bottom_rows_zero and right_columns_zero:
                    self.density_matrix = self.density_matrix[
                        :new_dimensions, :new_dimensions
                    ]
        self.dimensions = new_dimensions

    def set_index(self, minor, major=-1):
        if major >= 0:
            self.index = (major, minor)
        else:
            self.index = minor

    def measure(self, non_destructive=False, remove_composite=True, partial=False):
        if self.measured:
            raise FockAlreadyMeasuredException()
        outcome = None
        if isinstance(self.index, int):
            return self.envelope.measure(remove_composite=remove_composite)
        else:
            match self.expansion_level:
                case ExpansionLevel.Label:
                    outcome = self.label
                case ExpansionLevel.Vector:
                    probabilities = np.abs(self.state_vector.flatten()) ** 2
                    outcome = np.random.choice(len(probabilities), p=probabilities)
                case ExpansionLevel.Matrix:
                    probabilities = np.real(np.diag(self.density_matrix))
                    outcome = np.random.choice(len(probabilities), p=probabilities)
        if not partial:
            self.envelope._set_measured(remove_composite=remove_composite)
        self._set_measured()
        return outcome

    def _set_measured(self, **kwargs):
        self.measured = True
        self.label = None
        self.expansion_level = None
        self.state_vector = None
        self.density_matrix = None
        self.index = None

    def get_subspace(self):
        """
        Returns the space
        """
        print("GET SUBSPACE")
        if self.index is None:
            if not self.label is None:
                self.expand()

            if not self.state_vector is None:
                return self.state_vector
            elif not self.density_matrix is None:
                return self.density_matrix
        elif len(self.index) == 1:
            # State is in the Envelope
            pass

        elif len(self.index) == 2:
            state = self.envelope.composite_envelope._trace_out(self, destructive=False)
            return state


class FockAlreadyMeasuredException(Exception):
    pass
