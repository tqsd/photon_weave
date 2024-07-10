import random
import unittest

import numpy as np
from scipy.integrate import quad

from photon_weave.constants import gaussian
from photon_weave.operation.fock_operation import FockOperation, FockOperationType
from photon_weave.state.envelope import Envelope, TemporalProfile
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization, PolarizationLabel


class TestFock(unittest.TestCase):
    def test_envelope_initiation(self):
        envelope = Envelope()
        self.assertIsNone(envelope.composite_vector)
        self.assertIsNone(envelope.composite_matrix)
        self.assertEqual(envelope.fock, Fock())

        fock = Fock()
        fock.label = 1
        polarization = Polarization(PolarizationLabel.R)
        custom_envelope = Envelope(fock=fock, polarization=polarization)
        self.assertEqual(custom_envelope.polarization, polarization)
        self.assertEqual(custom_envelope.fock, fock)

    def test_envelope_combine(self):
        envelope = Envelope()
        envelope.combine()
        aux_fock = Fock()
        aux_fock.expand()
        aux_pol = Polarization()
        aux_pol.expand()

        expected_vector = np.kron(aux_fock.state_vector, aux_pol.state_vector)
        self.assertTrue(np.array_equal(envelope.composite_vector, expected_vector))
        self.assertEqual(envelope.fock.index, 0)
        self.assertEqual(envelope.polarization.index, 1)
        self.assertIsNone(envelope.fock.label)
        self.assertIsNone(envelope.fock.state_vector)
        self.assertIsNone(envelope.fock.density_matrix)
        self.assertIsNone(envelope.polarization.label)
        self.assertIsNone(envelope.polarization.state_vector)
        self.assertIsNone(envelope.polarization.density_matrix)

        envelope = Envelope()
        envelope.fock.expand()
        envelope.fock.expand()
        envelope.polarization.expand()

    def test_measure_envelope_fock(self):
        """
        Test measurement, where the fock state is not
        combined into an envelope or composite envelope.
        """
        r = random.randint(1, 20)
        env = Envelope()
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        env.apply_operation(op)
        outcome = env.measure()
        self.assertEqual(outcome, r)
        self.assertEqual(env.measured, True)
        self.assertEqual(env.fock.measured, True)
        self.assertEqual(env.fock.label, None)

        env = Envelope()
        env.fock.expand()
        env.apply_operation(op)
        outcome = env.measure()
        self.assertEqual(outcome, r)
        self.assertEqual(env.measured, True)
        self.assertEqual(env.fock.measured, True)
        self.assertEqual(env.fock.state_vector, None)

        env = Envelope()
        env.fock.expand()
        env.fock.expand()
        env.apply_operation(op)
        outcome = env.measure()
        self.assertEqual(outcome, r)
        self.assertEqual(env.measured, True)
        self.assertEqual(env.fock.measured, True)
        self.assertEqual(env.fock.density_matrix, None)

    def test_envelope_measurement(self):
        """
        Test measurement, when state is combined into an envelope
        """
        r = random.randint(1, 10)
        env = Envelope()
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        env.apply_operation(op)
        env.combine()
        outcome = env.measure()
        self.assertEqual(outcome, r)
        self.assertEqual(env.measured, True)
        self.assertEqual(env.composite_vector, None)
        self.assertEqual(env.composite_matrix, None)
        self.assertEqual(env.polarization.measured, True)
        self.assertEqual(env.polarization.label, None)
        self.assertEqual(env.polarization.state_vector, None)
        self.assertEqual(env.polarization.density_matrix, None)
        self.assertEqual(env.fock.measured, True)
        self.assertEqual(env.fock.label, None)
        self.assertEqual(env.fock.state_vector, None)
        self.assertEqual(env.fock.density_matrix, None)

    def test_envelope_overlap(self):
        env1 = Envelope()
        env2 = Envelope()
        f = env1.overlap_integral(env2, 0)


def two_gaussian_integral(sigma_a, sigma_b, t_a, t_b, omega_a, omega_b):
    tmp1 = -((t_a - t_b) ** 2) / (4 * sigma_a * sigma_b)
    tmp2 = -(sigma_a**2 * sigma_b**2 * (omega_a - omega_b) ** 2) / (
        sigma_a**2 + sigma_b**2
    )
    return np.exp(tmp1) * np.exp(tmp2)


class TestTemporalProfile(unittest.TestCase):
    def test_gaussian_integrals_perfect_overlap(self):
        """
        Testing the overlap of gaussian integrals
        """

        f1 = lambda x: gaussian(x, t_a=0, omega=100, mu=0, sigma=1)
        f2 = lambda x: gaussian(x, t_a=0, omega=100, mu=0, sigma=1)

        # NORM INTEGRAL
        norm1_integral, _ = quad(lambda x: np.abs(f1(x)) ** 2, -np.inf, np.inf)
        norm2_integral, _ = quad(lambda x: np.abs(f2(x)) ** 2, -np.inf, np.inf)

        integrand = lambda x: np.conj(f1(x)) * f2(x)
        result, error = quad(integrand, -np.inf, np.inf)
        # Normalization
        overlap = result / np.sqrt(norm1_integral * norm2_integral)

        a = two_gaussian_integral(1, 1, 0, 0, 100, 100)

        self.assertAlmostEqual(overlap, 1.0, places=6)

    def test_temporal_profile_partial_overlap(self):
        """
        Testing perfect overlap
        """

        f1 = lambda x: gaussian(x, t_a=0, omega=100, mu=0, sigma=1)
        f2 = lambda x: gaussian(x, t_a=1, omega=100, mu=0, sigma=1)

        # NORM INTEGRAL
        norm1_integral, _ = quad(lambda x: np.abs(f1(x)) ** 2, -np.inf, np.inf)
        norm2_integral, _ = quad(lambda x: np.abs(f2(x)) ** 2, -np.inf, np.inf)

        norm = norm1_integral * norm2_integral

        integrand = lambda x: np.conj(f1(x)) * f2(x)
        result, error = quad(integrand, -np.inf, np.inf)
        # Normalization
        overlap = result / np.sqrt(norm1_integral * norm2_integral)
        print(overlap)
        a = two_gaussian_integral(1, 1, 0, 1, 100, 100)
        print(a)

        self.assertAlmostEqual(overlap, 0.7788007, places=6)


if __name__ == "__main__":
    unittest.main()
