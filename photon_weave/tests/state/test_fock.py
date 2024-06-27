"""
Tests for the Fock class
"""
import random
import unittest

import numpy as np

from photon_weave.operation.fock_operation import FockOperation, FockOperationType
from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization


class TestFock(unittest.TestCase):
    def test_fock_initiation(self):
        fock = Fock()
        self.assertEqual(fock.label, 0)
        self.assertIsNone(fock.index)
        self.assertIsNone(fock.state_vector)
        self.assertIsNone(fock.density_matrix)

    def test_fock_expansion(self):
        fock = Fock()
        self.assertEqual(repr(fock), "|0⟩")
        fock.expand()
        expected_state_vector = np.zeros(3)
        expected_state_vector[0] = 1
        self.assertTrue(
            np.array_equal(fock.state_vector, expected_state_vector.reshape(-1, 1))
        )
        self.assertIsNone(fock.label)
        self.assertIsNone(fock.density_matrix)
        fock.expand()
        expected_density_matrix = np.outer(
            expected_state_vector.flatten(), np.conj(expected_state_vector.flatten())
        )
        self.assertTrue(np.array_equal(fock.density_matrix, expected_density_matrix))
        self.assertIsNone(fock.label)
        self.assertIsNone(fock.state_vector)

    def test_fock_eq(self):
        fock = Fock()
        fock_cmp = Fock()
        fock_false = Fock()
        fock_false.label = 2
        polarization = Polarization()
        self.assertFalse(fock == polarization)
        self.assertFalse(fock == fock_false)
        self.assertTrue(fock == fock_cmp)
        fock.expand()
        fock_cmp.expand()
        fock_false.expand()
        self.assertFalse(fock == fock_false)
        self.assertTrue(fock == fock_cmp)
        fock.expand()
        fock_cmp.expand()
        fock_false.expand()
        self.assertFalse(fock == fock_false)
        self.assertTrue(fock == fock_cmp)
        fock_other = Fock()
        self.assertFalse(fock == fock_other)

    def test_fock_repr(self):
        fock = Fock()
        self.assertEqual(repr(fock), "|0⟩")
        fock.expand()
        expected_state_vector = np.zeros(3)
        expected_state_vector[0] = 1
        expected_repr = "\n".join(
            [
                f"{complex_num.real:.2f} {'+' if complex_num.imag >= 0 else '-'} {abs(complex_num.imag):.2f}j"
                for complex_num in expected_state_vector
            ]
        )
        self.assertEqual(repr(fock), expected_repr)
        fock.expand()
        expected_density_matrix = np.outer(
            expected_state_vector.flatten(), np.conj(expected_state_vector.flatten())
        )

        expected_repr = "\n".join(
            [
                "\t".join(
                    [
                        f"({num.real:.2f} {'+' if num.imag >= 0 else '-'} {abs(num.imag):.2f}j)"
                        for num in row
                    ]
                )
                for row in expected_density_matrix
            ]
        )
        self.assertEqual(repr(fock), expected_repr)

        fock.density_matrix = None
        self.assertEqual(repr(fock), "Invalid Fock object")

        fock.index = 0
        self.assertEqual(repr(fock), "System is part of the Envelope")

    def test_expansion_level(self):
        fock = Fock()
        print(fock.expansion_level)
        self.assertEqual(fock.expansion_level, 0)
        fock.expand()
        self.assertEqual(fock.expansion_level, 1)
        fock.expand()
        self.assertEqual(fock.expansion_level, 2)

    def test_dimension_change(self):
        fock = Fock()
        fock.expand()
        expected_vector = np.array([1, 0, 0])
        self.assertTrue(
            np.array_equal(fock.state_vector, expected_vector.reshape(-1, 1))
        )
        fock.resize(10)
        expected_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(
            np.array_equal(fock.state_vector, expected_vector.reshape(-1, 1))
        )

        fock.resize(2)
        expected_vector = np.array([1, 0])
        self.assertTrue(
            np.array_equal(fock.state_vector, expected_vector.reshape(-1, 1))
        )

        fock.expand()

        expected_vector = np.array([1, 0])
        expected_density_matrix = np.outer(
            expected_vector.flatten(), np.conj(expected_vector.flatten())
        )
        self.assertTrue(np.array_equal(fock.density_matrix, expected_density_matrix))
        fock.resize(5)
        expected_vector = np.array([1, 0, 0, 0, 0])
        expected_density_matrix = np.outer(
            expected_vector.flatten(), np.conj(expected_vector.flatten())
        )
        self.assertTrue(np.array_equal(fock.density_matrix, expected_density_matrix))

        fock.resize(3)
        expected_vector = np.array([1, 0, 0])
        expected_density_matrix = np.outer(
            expected_vector.flatten(), np.conj(expected_vector.flatten())
        )
        self.assertTrue(np.array_equal(fock.density_matrix, expected_density_matrix))

    def test_measurement(self):
        r = random.randint(1, 10)
        env = Envelope()
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        env.apply_operation(op)
        outcome = env.fock.measure()
        self.assertEqual(outcome, r)
        self.assertEqual(env.measured, True)


# This allows the tests to be run with the script via the command line.
if __name__ == "__main__":
    unittest.main()
