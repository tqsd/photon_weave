import unittest

import numpy as np

from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization, PolarizationLabel


class TestPolarization(unittest.TestCase):
    def test_polarization_initiation(self):
        polarization = Polarization()
        self.assertIsNone(polarization.index)
        self.assertIsNone(polarization.state_vector)
        self.assertIsNone(polarization.density_matrix)
        self.assertIsNone(polarization.envelope)
        self.assertTrue(polarization.label == PolarizationLabel.H)

        for label in PolarizationLabel:
            p = Polarization(polarization=label)
            self.assertEqual(p.label, label)

    def test_polarization_expansion(self):
        for label in PolarizationLabel:
            p = Polarization(polarization=label)
            self.assertTrue(p.label == label)
            match label:
                case PolarizationLabel.H:
                    expected_state_vector = np.array([1 + 0j, 0 + 0j])
                case PolarizationLabel.V:
                    expected_state_vector = np.array([0 + 0j, 1 + 0j])
                case PolarizationLabel.R:
                    expected_state_vector = np.array(
                        [1 / np.sqrt(2) + 0j, 1j / np.sqrt(2)]
                    )
                case PolarizationLabel.L:
                    expected_state_vector = np.array(
                        [1 / np.sqrt(2) + 0j, -1j / np.sqrt(2)]
                    )
            p.expand()
            self.assertTrue(
                np.array_equal(p.state_vector, expected_state_vector.reshape(-1, 1))
            )
            self.assertIsNone(p.label)
            self.assertIsNone(p.index)
            self.assertIsNone(p.density_matrix)
            p.expand()
            self.assertIsNone(p.label)
            self.assertIsNone(p.index)
            self.assertIsNone(p.state_vector)
            expected_density_matrix = np.outer(
                expected_state_vector.flatten(),
                np.conj(expected_state_vector.flatten()),
            )
            self.assertTrue(np.array_equal(p.density_matrix, expected_density_matrix))


if __name__ == "__main__":
    unittest.main()
