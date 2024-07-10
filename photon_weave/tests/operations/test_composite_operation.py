"""
Test the Composite Operations
"""

import random
import unittest

import numpy as np

from photon_weave.operation.composite_operation import (
    CompositeOperation,
    CompositeOperationType,
)
from photon_weave.operation.fock_operation import FockOperation, FockOperationType
from photon_weave.operation.polarization_operations import (
    PolarizationOperation,
    PolarizationOperationType,
)
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization


class TestFockOperation(unittest.TestCase):
    def test_non_polarizing_beam_split_first(self):
        """
        Operator can be applied to the envlopes
        """
        np.set_printoptions(precision=4, threshold=1000)
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.resize(2)
        env2.fock.resize(2)
        op_create = FockOperation(operation=FockOperationType.Creation, apply_count=1)
        env2.fock.apply_operation(op_create)

        op = CompositeOperation(operation=CompositeOperationType.NonPolarizingBeamSplit)

        op.operate(env1, env2)

        expected_operator = np.zeros((4, 4), dtype=np.complex_)

        self.assertTrue(env1.composite_envelope is env2.composite_envelope)
        ce = env1.composite_envelope
        self.assertAlmostEqual(op.operator[0][0], 1)
        self.assertAlmostEqual(op.operator[1][1], 1 / np.sqrt(2))
        self.assertAlmostEqual(op.operator[1][2], 1j / np.sqrt(2))
        self.assertAlmostEqual(op.operator[2][1], 1j / np.sqrt(2))
        self.assertAlmostEqual(op.operator[2][2], 1 / np.sqrt(2))
        self.assertAlmostEqual(op.operator[3][3], 1)

    def test_non_polarizing_beam_splitter_second(self):
        """
        You can apply beam splitter to the composite envelope directly
        """

        e1 = Envelope()
        e2 = Envelope()
        create = FockOperation(FockOperationType.Creation)
        e1.apply_operation(create)
        c = CompositeEnvelope(e1, e2)
        bs = CompositeOperation(CompositeOperationType.NonPolarizingBeamSplit)
        c.apply_operation(bs, e1, e2)
        for i, v in enumerate(c.states[0][0]):
            if i == 1:
                self.assertAlmostEqual(v[0], 0.7071068j)
            elif i == 3:
                self.assertAlmostEqual(v[0], 0.7071068)
            else:
                self.assertAlmostEqual(v[0], 0)

    def test_non_polarizing_beam_splitter_third(self):
        """
        You can apply beam splitter to the composite envelope directly
        where the envelopes have different cutoffs
        """

        e1 = Envelope()
        e2 = Envelope()
        e1.fock.resize(5)
        e2.fock.resize(3)
        create = FockOperation(FockOperationType.Creation)
        e1.apply_operation(create)
        c = CompositeEnvelope(e1, e2)
        bs = CompositeOperation(CompositeOperationType.NonPolarizingBeamSplit)
        c.apply_operation(bs, e1, e2)
        for i, v in enumerate(c.states[0][0]):
            if i == 1:
                self.assertAlmostEqual(v[0], 0.707106781j)
            elif i == 3:
                self.assertAlmostEqual(v[0], 0.7071068)
            else:
                self.assertAlmostEqual(v[0], 0)

    def test_non_polarizing_beam_splitter_fourth(self):
        """
        Operator needs to be applied to a combined state
        """

        e1 = Envelope()
        e2 = Envelope()
        e1.fock.resize(5)
        e2.fock.resize(3)
        create = FockOperation(FockOperationType.Creation)
        e1.apply_operation(create)
        c = CompositeEnvelope(e1, e2)
        c.combine(e1.fock, e2.fock)
        bs = CompositeOperation(CompositeOperationType.NonPolarizingBeamSplit)
        c.apply_operation(bs, e1, e2)
        for i, v in enumerate(c.states[0][0]):
            if i == 1:
                self.assertAlmostEqual(v[0], 0.7071068j)
            elif i == 3:
                self.assertAlmostEqual(v[0], 0.7071068)
            else:
                self.assertAlmostEqual(v[0], 0)

    def test_cnot_operation(self):
        e1 = Envelope()
        e2 = Envelope()
        H = PolarizationOperation(operation=PolarizationOperationType.H)
        e1.apply_operation(H)
        CNOT = CompositeOperation(operation=CompositeOperationType.CNOT)
        CNOT.operate(e1, e2)
        states = e1.composite_envelope.states[0][0]
        self.assertAlmostEqual(1 / np.sqrt(2), states[0][0])
        self.assertAlmostEqual(0, states[1][0])
        self.assertAlmostEqual(0, states[2][0])
        self.assertAlmostEqual(1 / np.sqrt(2), states[3][0])

    def test_cnot_second(self):
        e1 = Envelope()
        e2 = Envelope()
        c = CompositeEnvelope(e1, e2)
        H = PolarizationOperation(operation=PolarizationOperationType.H)
        e1.apply_operation(H)
        CNOT = CompositeOperation(operation=CompositeOperationType.CNOT)
        c.apply_operation(CNOT, e1, e2)
        states = e1.composite_envelope.states[0][0]
        self.assertAlmostEqual(1 / np.sqrt(2), states[0][0])
        self.assertAlmostEqual(0, states[1][0])
        self.assertAlmostEqual(0, states[2][0])
        self.assertAlmostEqual(1 / np.sqrt(2), states[3][0])
