"""
Test the Fock Operation
"""

import random
import unittest

import numpy as np

from photon_weave.operation.fock_operation import FockOperation, FockOperationType
from photon_weave.operation.polarization_operations import (
    PolarizationOperation,
    PolarizationOperationType,
)
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization, PolarizationLabel


class TestPolarizationOperation(unittest.TestCase):
    def test_identity_operator(self):
        pol = Polarization()
        op = PolarizationOperation(PolarizationOperationType.I)
        pol.apply_operation(op)
        self.assertEqual(pol.label, PolarizationLabel.H)

        pol.expand()
        pol.apply_operation(op)
        self.assertTrue(np.array_equal(pol.state_vector, [[1], [0]]))

        pol.expand()
        pol.apply_operation(op)
        self.assertTrue(np.array_equal(pol.density_matrix, np.array([[1, 0], [0, 0]])))

    def test_X_operator(self):
        pol = Polarization()
        op = PolarizationOperation(PolarizationOperationType.X)
        pol.apply_operation(op)
        self.assertEqual(pol.label, PolarizationLabel.V)
        pol.apply_operation(op)
        self.assertEqual(pol.label, PolarizationLabel.H)
        pol = Polarization(PolarizationLabel.R)
        pol.apply_operation(op)
        self.assertEqual(pol.label, PolarizationLabel.L)
        pol = Polarization(PolarizationLabel.R)
        pol.apply_operation(op)
        self.assertEqual(pol.label, PolarizationLabel.L)

        pol = Polarization()
        pol.expand()
        pol.apply_operation(op)

        expected_vector = np.array([[0], [1]])
        self.assertTrue(np.array_equal(expected_vector, pol.state_vector))

        pol = Polarization(PolarizationLabel.R)
        pol.expand()
        pol.expand()
        pol.apply_operation(op)

        expected_vector = (1 / 2) * np.array([[1, 1j], [-1j, 1]])
        self.assertTrue(np.allclose(expected_vector, pol.density_matrix))

    def test_Y_operator(self):
        cases = [
            [PolarizationLabel.H, np.array([[0], [1j]])],
            [PolarizationLabel.V, np.array([[-1j], [0]])],
            [PolarizationLabel.R, np.array([[1 / np.sqrt(2)], [1j / np.sqrt(2)]])],
            [PolarizationLabel.L, np.array([[-1 / np.sqrt(2)], [1j / np.sqrt(2)]])],
        ]
        for c in cases:
            pol = Polarization(c[0])
            op = PolarizationOperation(PolarizationOperationType.Y)
            pol.apply_operation(op)
            self.assertTrue(
                np.allclose(c[1], np.array(pol.state_vector, dtype=np.complex_)),
                msg=f"Case {c[0]} doesn't produce correct state",
            )

    def test_envelope_operation(self):
        env = Envelope()
        op = PolarizationOperation(PolarizationOperationType.X)
        env.apply_operation(op)
        self.assertEqual(env.polarization.label, PolarizationLabel.V)

        env = Envelope()
        env.polarization.expand()
        env.combine()
        env.apply_operation(op)
        f = [[1], [0], [0]]
        p = [[0], [1]]
        v = np.kron(f, p)

        self.assertTrue(np.allclose(v, env.composite_vector))
