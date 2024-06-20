"""
Tests for the Fock class
"""
import random
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from photon_weave.operation.fock_operation import FockOperation, FockOperationType
from photon_weave.state.composite_envelope import (
    CompositeEnvelope,
    FockOrPolarizationExpectedException,
    StateNotInThisCompositeEnvelopeException,
    pad_operator,
)
from photon_weave.state.envelope import Envelope
from photon_weave.state.polarization import Polarization, PolarizationLabel


class TestFock(unittest.TestCase):
    def test_initiation_from_envelopes(self):
        env1 = Envelope()
        env2 = Envelope()
        composite_envelope = CompositeEnvelope(env1, env2)
        self.assertEqual(env1.composite_envelope, composite_envelope)
        self.assertEqual(env2.composite_envelope, composite_envelope)
        self.assertTrue(env1 in composite_envelope.envelopes)
        self.assertTrue(env2 in composite_envelope.envelopes)

    def test_envelope_addition(self):
        env1 = Envelope()
        env2 = Envelope()
        composite_envelope = CompositeEnvelope(env1, env2)
        env3 = Envelope()
        composite_envelope.add_envelope(env3)
        self.assertEqual(env3.composite_envelope, composite_envelope)
        self.assertTrue(env3 in composite_envelope.envelopes)

    def test_composite_envelope_merge(self):
        env1 = Envelope()
        env2 = Envelope()
        ce1 = CompositeEnvelope(env1, env2)

        env3 = Envelope()
        env4 = Envelope()
        ce2 = CompositeEnvelope(env3, env4)

        ce3 = CompositeEnvelope(ce1, ce2)

        self.assertEqual(len(ce3.envelopes), 4)
        for e in [env1, env2, env3, env4]:
            self.assertTrue(
                e in ce3.envelopes, msg="Added envelope should be in envelopes"
            )
            self.assertEqual(e.composite_envelope, ce3)

    def test_composite_envelope_combine(self):
        env1 = Envelope()
        env2 = Envelope()
        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.fock, env2.fock)
        ce1.combine(env1.polarization, env2.polarization)

        self.assertEqual(env1.fock.index, (0, 0))
        self.assertEqual(env2.fock.index, (0, 1))
        self.assertEqual(env1.polarization.index, (1, 0))
        self.assertEqual(env2.polarization.index, (1, 1))
        self.assertEqual(env1.composite_envelope, ce1)
        self.assertEqual(env2.composite_envelope, ce1)
        self.assertTrue(np.array_equal(ce1.states[1][0], [[1], [0], [0], [0]]))

        env3 = Envelope()
        ce1.add_envelope(env3)
        ce1.combine(env1.polarization, env3.polarization)
        self.assertTrue(
            np.array_equal(ce1.states[1][0], [[1], [0], [0], [0], [0], [0], [0], [0]])
        )
        self.assertEqual(env3.polarization.index, (1, 2))

    def test_composite_combine(self):
        env1 = Envelope()
        env2 = Envelope()
        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.fock, env2.fock)
        ce1.combine(env1.polarization, env2.polarization)

        env3 = Envelope()
        env4 = Envelope()
        ce2 = CompositeEnvelope(env3, env4)
        ce2.combine(env3.fock, env4.fock)
        ce2.combine(env3.polarization, env4.polarization)
        ce3 = CompositeEnvelope(ce1, ce2)
        self.assertTrue(np.array_equal(ce3.states[1][0], [[1], [0], [0], [0]]))
        self.assertTrue(np.array_equal(ce3.states[3][0], [[1], [0], [0], [0]]))
        ce3.combine(env1.polarization, env3.polarization)
        self.assertTrue(
            np.array_equal(
                ce3.states[1][0],
                [
                    [1],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                    [0],
                ],
            )
        )
        self.assertEqual(len(ce3.states), 3, msg="Is the residual vector deleted?")

    def test_matrix_combine(self):
        env1 = Envelope()
        env2 = Envelope()
        env1.polarization.expand()
        env1.polarization.expand()
        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.polarization, env2.polarization)
        self.assertTrue(
            np.array_equal(
                ce1.states[0][0],
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            )
        )

    def test_precombined_envelope(self):
        env1 = Envelope()
        env1.fock.expand()
        env1.combine()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        self.assertTrue(
            np.array_equal(
                ce.states[0][0],
                [[1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]],
            )
        )

        env3 = Envelope()
        env3.fock.expand()
        env3.fock.expand()
        env3.combine()
        env4 = Envelope()
        ce1 = CompositeEnvelope(env3, env4)
        ce2 = CompositeEnvelope(ce1, ce)
        ce2.combine(env3.fock, env1.polarization)
        expected_vector = np.zeros(72)
        expected_vector[0] = 1
        expected_density_matrix = np.outer(
            expected_vector.flatten(), np.conj(expected_vector.flatten())
        )
        self.assertTrue(np.array_equal(ce2.states[0][0], expected_density_matrix))

    def test_exception(self):
        with self.assertRaises(StateNotInThisCompositeEnvelopeException):
            env1 = Envelope()
            env2 = Envelope()
            ce = CompositeEnvelope(env1, env2)
            env3 = Envelope()
            ce.combine(env1.fock, env3.fock)
        with self.assertRaises(FockOrPolarizationExpectedException):
            env1 = Envelope()
            env2 = Envelope()
            ce = CompositeEnvelope(env1, env2)
            ce.combine(env1, env2)

    def test_composite_expansion(self):
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        ce.expand(env1.polarization)
        self.assertEqual(ce.states[0][0].shape[0], 4)
        self.assertEqual(ce.states[0][0].shape[1], 4)
        ce.expand(env1.polarization)

    def test_composite_reorder(self):
        p1 = Polarization(polarization=PolarizationLabel.V)
        p2 = Polarization(polarization=PolarizationLabel.H)
        env1 = Envelope(polarization=p1)
        env2 = Envelope(polarization=p2)
        env1.polarization.expand()
        env2.polarization.expand()
        state1 = env1.polarization.state_vector.copy()
        state2 = env2.polarization.state_vector.copy()

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        self.assertTrue(np.array_equal(ce.states[0][0], np.kron(state1, state2)))
        self.assertEqual(ce.states[0][1], [env1.polarization, env2.polarization])
        ce.rearange(env2.polarization, env1.polarization)
        self.assertTrue(np.array_equal(ce.states[0][0], np.kron(state2, state1)))
        self.assertEqual(ce.states[0][1], [env2.polarization, env1.polarization])
        ce.rearange(env1.polarization, env2.polarization)

    def test_composite_reorder_matrix(self):
        p1 = Polarization(polarization=PolarizationLabel.R)
        p1.expand()
        p1.expand()
        pol = p1.density_matrix.copy()
        env1 = Envelope(polarization=p1)
        env2 = Envelope()
        env1.fock.dimensions = 5
        env1.fock.expand()
        env1.fock.expand()
        f1 = env1.fock.density_matrix.copy()
        env2.fock.dimensions = 4
        env2.fock.expand()
        env2.fock.expand()
        f2 = env2.fock.density_matrix.copy()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock, env1.polarization)

        expected_matrix = np.kron(np.kron(f1, f2), pol)
        self.assertTrue(np.array_equal(expected_matrix, ce.states[0][0]))
        self.assertEqual(ce.states[0][1], [env1.fock, env2.fock, env1.polarization])
        ce.rearange(env1.polarization, env1.fock)

        expected_matrix = np.kron(np.kron(pol, f1), f2)
        self.assertTrue(np.array_equal(expected_matrix, ce.states[0][0]))
        self.assertEqual(ce.states[0][1], [env1.polarization, env1.fock, env2.fock])

    def test_fock_operations_on_composite_envelope_first(self):
        # Applying Fock operations to the composite envelope
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        op = FockOperation(FockOperationType.Creation)
        # Applying Fock operation to the fock state
        ce.apply_operation(op, env1.fock)
        self.assertEqual(env1.fock.label, 1)
        self.assertEqual(env1.polarization.label, PolarizationLabel.H)
        # Applying Fock operation to the envelope
        # It should be correctly routed
        ce.apply_operation(op, env1)
        self.assertEqual(env1.fock.label, 2)
        self.assertEqual(env1.polarization.label, PolarizationLabel.H)

        # Combining the envelope
        e1 = Envelope()
        e2 = Envelope()
        c = CompositeEnvelope(e1, e2)
        e1.combine()
        op = FockOperation(FockOperationType.Creation)
        c.apply_operation(op, e1)
        expected_vector = np.kron([[0], [1], [0]], [[1], [0]])
        self.assertTrue(np.array_equal(e1.composite_vector, expected_vector))
        # Combining the composite envelope
        e1 = Envelope()
        e2 = Envelope()
        c = CompositeEnvelope(e1, e2)
        c.combine(e1.fock, e2.fock, e1.polarization)
        op = FockOperation(FockOperationType.Creation)
        c.apply_operation(op, e1.fock)

        expected_vector = [[0], [1], [0]]
        expected_vector = np.kron(expected_vector, [[1], [0], [0]])
        expected_vector = np.kron(expected_vector, [[1], [0]])
        assert_array_equal(c.states[0][0], expected_vector)

    def test_composite_nested(self):
        env1 = Envelope()
        env2 = Envelope()

        ce1 = CompositeEnvelope(env1, env2)
        ce2 = CompositeEnvelope(env1, env2)
        self.assertEqual(ce1.states, [])
        try:
            # Trigger combination of states in ce1
            # eventhough the states are in ce2
            ce1.combine(env1.fock, env2.fock)
        except Exception as _:
            self.fail("ce1.combine() should work")

    def test_measurement_fock(self):
        """
        Measurement test of uncombined fock state
        """
        r = random.randint(1, 10)
        env1 = Envelope()
        env2 = Envelope()
        c = CompositeEnvelope(env2, env1)
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        c.apply_operation(op, env1)

        outcome = c.measure(env1)
        self.assertEqual(outcome[0], r)
        self.assertFalse(env1 in c.envelopes)
        self.assertEqual(env1.measured, True)
        self.assertEqual(env1.composite_envelope, None)
        self.assertEqual(env1.composite_vector, None)
        self.assertEqual(env1.composite_matrix, None)
        self.assertEqual(env1.fock.measured, True)
        self.assertEqual(env1.fock.label, None)
        self.assertEqual(env1.fock.state_vector, None)
        self.assertEqual(env1.fock.density_matrix, None)
        self.assertEqual(env1.polarization.measured, True)
        self.assertEqual(env1.polarization.label, None)
        self.assertEqual(env1.polarization.state_vector, None)
        self.assertEqual(env1.polarization.density_matrix, None)

    def test_measurement_envelope(self):
        """
        Measurement test
        State combined in envelope
        """
        r = random.randint(1, 4)
        env1 = Envelope()
        env2 = Envelope()
        c = CompositeEnvelope(env2, env1)
        env1.fock.dimensions = 5
        env1.combine()
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        c.apply_operation(op, env1)
        outcome = c.measure(env1)
        self.assertEqual(outcome[0], r)
        self.assertFalse(env1 in c.envelopes)
        self.assertEqual(env1.measured, True)
        self.assertEqual(env1.composite_envelope, None)
        self.assertEqual(env1.composite_vector, None)
        self.assertEqual(env1.composite_matrix, None)
        self.assertEqual(env1.fock.measured, True)
        self.assertEqual(env1.fock.label, None)
        self.assertEqual(env1.fock.state_vector, None)
        self.assertEqual(env1.fock.density_matrix, None)
        self.assertEqual(env1.polarization.measured, True)
        self.assertEqual(env1.polarization.label, None)
        self.assertEqual(env1.polarization.state_vector, None)
        self.assertEqual(env1.polarization.density_matrix, None)

    def test_measurement_envelope_matrix_form(self):
        """
        Measurement test
        State combined in envelope
        """
        r = random.randint(1, 4)
        env1 = Envelope()
        env2 = Envelope()
        c = CompositeEnvelope(env2, env1)
        env1.fock.dimensions = 5
        env1.fock.expand()
        env1.fock.expand()
        env1.combine()
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        c.apply_operation(op, env1)
        outcome = c.measure(env1)
        self.assertEqual(outcome, r)
        self.assertFalse(env1 in c.envelopes)
        self.assertEqual(env1.measured, True)
        self.assertEqual(env1.composite_envelope, None)
        self.assertEqual(env1.composite_vector, None)
        self.assertEqual(env1.composite_matrix, None)
        self.assertEqual(env1.fock.measured, True)
        self.assertEqual(env1.fock.label, None)
        self.assertEqual(env1.fock.state_vector, None)
        self.assertEqual(env1.fock.density_matrix, None)
        self.assertEqual(env1.polarization.measured, True)
        self.assertEqual(env1.polarization.label, None)
        self.assertEqual(env1.polarization.state_vector, None)
        self.assertEqual(env1.polarization.density_matrix, None)

    def test_measurement_envelope_matrix_form(self):
        """
        Measurement test
        State combined in envelope
        We try to measure the envelope
        """
        r = random.randint(1, 4)
        env1 = Envelope()
        env2 = Envelope()
        c = CompositeEnvelope(env2, env1)
        env1.fock.dimensions = 5
        env1.fock.expand()
        env1.fock.expand()
        env1.combine()
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        c.apply_operation(op, env1)
        outcome = env1.measure()
        self.assertEqual(outcome, r)
        self.assertFalse(env1 in c.envelopes)
        self.assertEqual(env1.measured, True)
        self.assertEqual(env1.composite_envelope, None)
        self.assertEqual(env1.composite_vector, None)
        self.assertEqual(env1.composite_matrix, None)
        self.assertEqual(env1.fock.measured, True)
        self.assertEqual(env1.fock.label, None)
        self.assertEqual(env1.fock.state_vector, None)
        self.assertEqual(env1.fock.density_matrix, None)
        self.assertEqual(env1.polarization.measured, True)
        self.assertEqual(env1.polarization.label, None)
        self.assertEqual(env1.polarization.state_vector, None)
        self.assertEqual(env1.polarization.density_matrix, None)

    def test_measurement_composite_combine_vector(self):
        """
        Verious tests of measurement when
        envelope state is combined in various composite
        product spaces in CompositeEvnelope
        """
        r = random.randint(1, 4)
        r = 2
        env1 = Envelope()
        env2 = Envelope()
        env3 = Envelope()
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        c = CompositeEnvelope(env1, env2, env3)
        c.apply_operation(op, env1)
        env1.fock.dimensions = 3
        env2.fock.dimensions = 3
        env3.fock.dimensions = 3
        c.combine(env1.fock, env2.fock, env3.fock)
        m = c.measure(env1)
        self.assertEqual(r, m[0])
        self.assertFalse(env1.fock in c.states[0][1])
        self.assertIsNone(env1.composite_vector)
        self.assertIsNone(env1.composite_matrix)
        self.assertIsNone(env1.fock.index)
        self.assertTrue(env1.fock.measured)
        self.assertIsNone(env1.fock.label)
        self.assertIsNone(env1.fock.state_vector)
        self.assertIsNone(env1.fock.density_matrix)
        self.assertFalse(env1 in c.envelopes)
        self.assertTrue(env2 in c.envelopes)
        self.assertTrue(env3 in c.envelopes)
        expected_vector = np.zeros(9)
        expected_vector[0] = 1
        expected_vector = expected_vector.reshape(-1, 1)
        self.assertTrue(np.array_equal(c.states[0][0], expected_vector))

    def test_polarization_measurement(self):
        """
        If polarization is included in the same space, it should also be removed
        """
        env1 = Envelope()
        env2 = Envelope()
        c = CompositeEnvelope(env1, env2)
        r = random.randint(1, 5)
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        c.apply_operation(op, env1)
        c.combine(env1.polarization, env1.fock, env2.fock)

    def test_measurement_removal(self):
        """
        Test if the envelope is uncombined if only one state is combined
        """
        env1 = Envelope()
        env2 = Envelope()
        c = CompositeEnvelope(env1, env2)
        r = random.randint(1, 2)
        r = 1
        op = FockOperation(FockOperationType.Creation, apply_count=r)
        c.apply_operation(op, env1)
        env1.fock.expand()
        env2.fock.expand()
        c.combine(env1.fock, env2.fock)

        m = c.measure(env1)

    def test_trace_out(self):
        """
        Testing if the traceout
        """
        env1 = Envelope()
        env2 = Envelope()
        env3 = Envelope()
        env1.fock.label = 1
        env2.fock.label = 3
        env3.fock.label = 1
        env1.fock.dimensions = 2
        env2.fock.dimensions = 5
        env3.fock.dimensions = 3
        c = CompositeEnvelope(env1, env2, env3)
        c.combine(env2.fock, env1.fock, env3.fock)
        self.assertEqual(len(c.states[0][0]), 2 * 5 * 3)
        c._trace_out(env1.fock)
        self.assertEqual(len(c.states[0][0]), 5 * 3)
        e2 = np.array([[0], [0], [0], [1], [0]], dtype=np.float_)
        e3 = np.array([[0], [1], [0]])
        exp = np.kron(e2, e3)
        x = exp - c.states[0][0]
        self.assertTrue(np.array_equal(c.states[0][0], np.kron(e2, e3)))

    def test_automatic_polarization_measurement(self):
        """
        Test the automatic measurement of polarization
        when fock state is measured
        """

        env1 = Envelope(polarization=Polarization(PolarizationLabel.V))
        env2 = Envelope()
        c = CompositeEnvelope(env1, env2)
        c.combine(env1.polarization, env2.fock, env2.polarization)
        m = c.measure(env1)
        self.assertEqual(m[0], 0)
        self.assertIsNone(env1.polarization.label)
        self.assertIsNone(env1.polarization.state_vector)
        self.assertIsNone(env1.polarization.density_matrix)
        self.assertIsNone(env1.fock.label)
        self.assertIsNone(env1.fock.state_vector)
        self.assertIsNone(env1.fock.density_matrix)
        self.assertIsNone(env1.composite_envelope)
        self.assertFalse(env1.polarization in c.states[0][1])

    def test_matrix_measurement(self):
        env1 = Envelope(polarization=Polarization(PolarizationLabel.V))
        env2 = Envelope()
        env1.fock.label = 1
        env1.fock.expand()
        env1.fock.expand()
        env1.polarization.expand()
        env1.polarization.expand()
        c = CompositeEnvelope(env1, env2)
        c.combine(env1.polarization, env2.fock, env2.polarization)
        m = c.measure(env1)
        self.assertEqual(1, m[0])
        self.assertFalse(env1 in c.envelopes)
        self.assertFalse(env1.polarization in c.states[0][1])
        self.assertFalse(env1.fock in c.states[0][1])

    def test_POVM_measurement(self):
        """
        Test the POVM measurement ensuring that it handles state and operator extension,
        correct probability calculations, and state updates correctly.
        """
        # Create envelopes with dummy states that have simple behaviors
        env1 = Envelope(polarization=Polarization(PolarizationLabel.H))
        env2 = Envelope(polarization=Polarization(PolarizationLabel.V))

        # Initialize composite envelope
        c = CompositeEnvelope(env1, env2)

        # Define properly summing up to identity POVM operators for a two-state system
        E1 = np.array([[2 / 3, 0], [0, 0]])  # Acts more on |0>
        E2 = np.array([[1 / 3, 0], [0, 1]])  # Acts partially on |0> and fully on |1>

        # Combine states into the composite envelope
        c.combine(env1.polarization, env2.polarization)

        # Perform POVM measurement with the defined operators
        result_index = c.POVM_measurement(
            [env1.polarization, env2.polarization], [E1, E2]
        )

        post_state = c.states[0][0]

        self.assertIn(result_index, [0, 1], "Outcome should be either 0 or 1.")
        norm_post_state = np.linalg.norm(post_state)
        self.assertTrue(
            np.isclose(norm_post_state, 1),
            "Post-measurement state is not normalized correctly.",
        )

    def test_POVM_measurement_three_envelopes(self):
        env1 = Envelope(polarization=Polarization(PolarizationLabel.H))
        env2 = Envelope(polarization=Polarization(PolarizationLabel.V))
        env3 = Envelope(polarization=Polarization(PolarizationLabel.H))

        c = CompositeEnvelope(env1, env2, env3)

        # Define POVM operators that properly sum to identity
        E1 = np.array([[0.7, 0], [0, 0.2]])
        E2 = np.array([[0.2, 0], [0, 0.7]])
        E3 = np.eye(2) - E1 - E2  # Ensure completeness relation

        c.combine(env1.polarization, env2.polarization, env3.polarization)

        result_index = c.POVM_measurement(
            [env1.polarization, env2.polarization],
            [E1, E2, E3],  # Include E3 to cover the entire space
        )

        post_state = c.states[0][0]
        self.assertIn(result_index, [0, 1, 2], "Outcome index should be 0, 1, or 2.")

        norm_post_state = np.linalg.norm(post_state)
        self.assertTrue(
            np.isclose(norm_post_state, 1),
            "Post-measurement state is not normalized correctly.",
        )

    def test_POVM_measurement_three_envelopes_matrix(self):
        env1 = Envelope(polarization=Polarization(PolarizationLabel.H))
        env2 = Envelope(polarization=Polarization(PolarizationLabel.V))
        env3 = Envelope(polarization=Polarization(PolarizationLabel.H))
        env1.fock.expand()
        env1.fock.expand()

        c = CompositeEnvelope(env1, env2, env3)

        # Define POVM operators that properly sum to identity
        E1 = np.array([[0.7, 0], [0, 0.2]])
        E2 = np.array([[0.2, 0], [0, 0.7]])
        E3 = np.eye(2) - E1 - E2  # Ensure completeness relation

        c.combine(env1.polarization, env2.polarization, env3.polarization)

        result_index = c.POVM_measurement(
            [env1.polarization, env2.polarization],
            [E1, E2, E3],  # Include E3 to cover the entire space
        )

        post_state = c.states[0][0]
        self.assertIn(result_index, [0, 1, 2], "Outcome index should be 0, 1, or 2.")

        norm_post_state = np.linalg.norm(post_state)
        self.assertTrue(
            np.isclose(norm_post_state, 1),
            "Post-measurement state is not normalized correctly.",
        )

    def known_POVM_test(self):
        env1 = Envelope(polarization=Polarization(PolarizationLabel.H))
        env2 = Envelope(polarization=Polarization(PolarizationLabel.V))
        env1.polarization.expand()
        env1.polarization.expand()

        c = CompositeEnvelope(env1, env2)
        c.combine(env1.polarization, env2.polarization)

        E1 = np.array([[1 / 3, 0], [0, 0]])
        E2 = np.array([[0, 0], [0, 1 / 3]])
        E3 = np.eye(2) - E1 - E2

        c.POVM_measurement([env1.polarization], [E1, E2, E3])


class TestPadOperator(unittest.TestCase):
    def test_pad_operator_single(self):
        # Test padding an operator for a single subsystem
        state_dims = [2, 2, 2]  # Three qubits
        operator = np.array([[1, 0], [0, 0]])  # Projection operator on the first qubit
        target_index = 0
        expected_result = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        padded_op = pad_operator(operator, state_dims, target_index)
        np.testing.assert_array_equal(padded_op, expected_result)

    def test_pad_operator_span_multiple(self):
        # Test padding an operator that spans multiple subsystems
        state_dims = [2, 2, 2]  # Three qubits
        operator = np.eye(4)  # Identity on two qubits
        target_index = 1
        padded_op = pad_operator(operator, state_dims, target_index)
        expected_result = np.kron(np.eye(2), np.eye(4))
        np.testing.assert_array_equal(padded_op, expected_result)

    def test_operator_dimension_mismatch(self):
        # Test handling of dimension mismatch
        state_dims = [2, 3, 2]
        operator = np.eye(2)  # Operator intended for a single qubit
        target_index = 1
        with self.assertRaises(ValueError):
            pad_operator(operator, state_dims, target_index)
