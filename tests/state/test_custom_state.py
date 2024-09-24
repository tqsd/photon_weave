import unittest

import jax.numpy as jnp

from photon_weave.photon_weave import Config
from photon_weave.state.custom_state import CustomState
from photon_weave.state.expansion_levels import ExpansionLevel


class TestCustomStateExpansionContraction(unittest.TestCase):
    def test_custom_state_expand(self) -> None:
        cs = CustomState(3)
        cs.expand()
        self.assertEqual(cs.expansion_level, ExpansionLevel.Vector)
        self.assertTrue(jnp.allclose(jnp.array([[1], [0], [0]]), cs.state))
        cs.expand()
        self.assertEqual(cs.expansion_level, ExpansionLevel.Matrix)
        self.assertTrue(
            jnp.allclose(jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), cs.state)
        )
        cs.contract(final=ExpansionLevel.Vector)
        self.assertEqual(cs.expansion_level, ExpansionLevel.Vector)
        self.assertTrue(jnp.allclose(jnp.array([[1], [0], [0]]), cs.state))
        cs.contract(final=ExpansionLevel.Label)
        self.assertEqual(cs.expansion_level, ExpansionLevel.Label)
        self.assertEqual(cs.state, 0)

    def test_contraction_to_label(self):
        cs = CustomState(10)
        cs.state = 8
        cs.expand()
        self.assertEqual(cs.expansion_level, ExpansionLevel.Vector)
        cs.expand()
        self.assertEqual(cs.expansion_level, ExpansionLevel.Matrix)
        cs.contract()
        self.assertEqual(cs.state, 8)
        self.assertEqual(cs.expansion_level, ExpansionLevel.Label)


class TestCusomStateMeasurement(unittest.TestCase):
    def test_all_measurement(self):
        for i in range(9):
            cs = CustomState(10)
            cs.state = i
            outcome = cs.measure()
            self.assertEqual(outcome[cs], i)
            self.assertEqual(cs.state, i)
            self.assertEqual(cs.expansion_level, ExpansionLevel.Label)
            cs.expand()
            outcome = cs.measure()
            self.assertEqual(outcome[cs], i)
            self.assertEqual(cs.state, i)
            self.assertEqual(cs.expansion_level, ExpansionLevel.Label)
            cs.expand()
            cs.expand()
            outcome = cs.measure()
            self.assertEqual(outcome[cs], i)
            self.assertEqual(cs.state, i)
            self.assertEqual(cs.expansion_level, ExpansionLevel.Label)

    def test_measure_POVM(self):
        C = Config()
        C.set_contraction(True)
        cs = CustomState(2)
        cs.state = 1
        operators = []
        for i in range(2):
            tmp_op = jnp.zeros((2, 2))
            tmp_op = tmp_op.at[i, i].set(1)
            operators.append(tmp_op)
        outcome = cs.measure_POVM(operators)
        self.assertEqual(outcome[0], 1)
        self.assertEqual(cs.expansion_level, ExpansionLevel.Label)
        self.assertEqual(cs.state, 1)


class TestCustomStateKrauApply(unittest.TestCase):
    def test_kraus_apply(self) -> None:
        cs = CustomState(2)
        op1 = jnp.array([[0, 0], [1, 0]])
        op2 = jnp.array([[0, 0], [0, 1]])
        cs.apply_kraus([op1, op2])
        self.assertEqual(cs.state, 1)

    def test_kraus_apply_exception_dimensions(self) -> None:
        cs = CustomState(2)
        op1 = jnp.array([[0, 0, 0], [1, 0, 0]])
        op2 = jnp.array([[0, 0], [0, 1]])
        with self.assertRaises(ValueError) as context:
            cs.apply_kraus([op1, op2])

    def test_kraus_apply_exception_identity(self) -> None:
        cs = CustomState(2)
        op1 = jnp.array([[0, 0], [1, 0]])
        op2 = jnp.array([[1, 0], [0, 0]])
        with self.assertRaises(ValueError) as context:
            cs.apply_kraus([op1, op2])
