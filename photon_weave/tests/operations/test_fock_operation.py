"""
Test the Fock Operation
"""
import random
import unittest

import numpy as np

from photon_weave.operation.fock_operation import FockOperation, FockOperationType
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization


class TestFockOperation(unittest.TestCase):
    def test_init(self):
        for t in FockOperationType:
            r = random.randint(0, 100)
            op = FockOperation(operation=t, apply_count=r, phi=0, alpha=0, zeta=0)
            self.assertEqual(r, op.apply_count)
            self.assertEqual(t, op.operation)

    def test_creation_operation(self):
        fock = Fock()
        op = FockOperation(FockOperationType.Creation)
        fock.apply_operation(operation=op)
        self.assertEqual(fock.label, 1)

        fock.expand()
        fock.apply_operation(operation=op)
        expected_vector = np.array([0, 0, 1, 0])
        self.assertTrue(
            np.array_equal(fock.state_vector, expected_vector.reshape(-1, 1))
        )

        fock.expand()
        fock.apply_operation(operation=op)
        expected_vector = np.array([0, 0, 0, 1])
        expected_density_matrix = np.outer(
            expected_vector.flatten(), np.conj(expected_vector.flatten())
        )
        self.assertTrue(np.array_equal(fock.density_matrix, expected_density_matrix))

        fock.apply_operation(operation=op)
        expected_vector = np.array([0, 0, 0, 0, 1])
        expected_density_matrix = np.outer(
            expected_vector.flatten(), np.conj(expected_vector.flatten())
        )
        self.assertTrue(np.array_equal(fock.density_matrix, expected_density_matrix))
        op = FockOperation(FockOperationType.Creation, apply_count=3)
        fock.apply_operation(operation=op)
        expected_vector = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        expected_density_matrix = np.outer(
            expected_vector.flatten(), np.conj(expected_vector.flatten())
        )
        self.assertTrue(np.array_equal(fock.density_matrix, expected_density_matrix))

    def test_creation_operator_second(self):
        """
        Making sure that the creation operator is correctly applied to the state which is in envelop
        or in composite envelope
        """
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.resize(2)
        env2.fock.resize(2)
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock)
        op = FockOperation(operation=FockOperationType.Creation)
        ce._apply_operator(op, env1.fock)
        self.assertTrue(np.array_equal(ce.states[0][0], [[0], [0], [1], [0]]))

    def test_annihilation_operator(self):
        fock = Fock()
        op = FockOperation(FockOperationType.Annihilation)
        fock.apply_operation(operation=op)
        self.assertEqual(fock.label, 0)

        op_create = FockOperation(FockOperationType.Creation, apply_count=3)
        fock.apply_operation(operation=op_create)
        self.assertEqual(fock.label, 3)

        fock.apply_operation(operation=op)
        self.assertEqual(fock.label, 2)

        fock.expand()
        fock.apply_operation(operation=op)

        expected_vector = np.array([0, 1, 0, 0, 0])
        self.assertTrue(
            np.array_equal(fock.state_vector, expected_vector.reshape(-1, 1))
        )

        expected_vector = np.array([1, 0, 0, 0, 0])
        fock.apply_operation(op)
        self.assertTrue(
            np.array_equal(fock.state_vector, expected_vector.reshape(-1, 1))
        )
        op_create = FockOperation(FockOperationType.Creation, apply_count=5)
        fock.expand()
        fock.apply_operation(op_create)
        expected_vector = np.array([0, 0, 0, 0, 0, 1], dtype=np.complex_)
        expected_density_matrix = np.outer(
            expected_vector.flatten(), np.conj(expected_vector.flatten())
        )

        # Check if there are differences greater than the tolerance
        self.assertTrue(
            np.allclose(
                fock.density_matrix, expected_density_matrix, atol=1e-10, rtol=1e-4
            )
        )
        op_annihilate = FockOperation(FockOperationType.Annihilation, apply_count=5)
        fock.apply_operation(op_annihilate)

        expected_vector = np.array([1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j])
        expected_density_matrix = np.outer(
            expected_vector.flatten(), np.conj(expected_vector.flatten())
        )

        self.assertTrue(np.allclose(fock.density_matrix, expected_density_matrix))

    def test_phase_shift_operator(self):
        with self.assertRaises(KeyError) as context:
            op = FockOperation(FockOperationType.PhaseShift, apply_count=1)

        fock = Fock()
        op = FockOperation(FockOperationType.PhaseShift, phi=1)
        fock.apply_operation(op)
        expected_vector = np.array([1, 0, 0])
        self.assertTrue(
            np.array_equal(fock.state_vector, expected_vector.reshape(-1, 1))
        )

        def create_operator(phi, cutoff):
            n = np.arange(cutoff)
            phases = np.exp(1j * n * phi)
            phase_shift_operator = np.diag(phases)
            return phase_shift_operator

        op_create = FockOperation(FockOperationType.Creation, 1)
        fock.apply_operation(op_create)
        fock.apply_operation(op)

        expected_vector = np.array([0, 1, 0])
        expected_vector = expected_vector.reshape(-1, 1)
        expected_vector = create_operator(1, 3) @ expected_vector
        self.assertTrue(np.array_equal(fock.state_vector, expected_vector))

    def test_displace_operator(self):
        fock = Fock()
        op = FockOperation(FockOperationType.Displace, alpha=2)
        fock.apply_operation(op)

    def test_squeeze_operator(self):
        fock = Fock()
        op = FockOperation(FockOperationType.Squeeze, zeta=0 + 1j)
        fock.apply_operation(op)

    def test_custom_operator(self):
        fock = Fock()
        fock.label = 1
        op = FockOperation(
            FockOperationType.Custom, expression=("expm", ("s_mult", -1j, np.pi, "n"))
        )
        fock.apply_operation(op)

        expected_vector = np.array([0.0 + 0j, -1.0 + 0j, 0.0 + 0j, 0 + 0j])
        expected_vector = expected_vector.reshape(-1, 1)
        self.assertTrue(
            np.allclose(
                fock.state_vector,
                expected_vector,
                atol=1e-10,  # Absolute tolerance
                rtol=0,  # Relative tolerance
            )
        )
