import unittest

import jax.numpy as jnp
import pytest

from photon_weave.operation import CustomStateOperationType, Operation
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.custom_state import CustomState


class TestCustomCustomOperation(unittest.TestCase):
    def test_custom_custom_vector(self) -> None:
        cs = CustomState(3)
        operator = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        op = Operation(CustomStateOperationType.Custom, operator=operator)
        cs.apply_operation(op)
        self.assertEqual(cs.state, 1)

    def test_custom_custom_matrix(self) -> None:
        cs = CustomState(3)
        cs.expand()
        cs.expand()
        operator = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        op = Operation(CustomStateOperationType.Custom, operator=operator)
        cs.apply_operation(op)
        self.assertEqual(cs.state, 1)

    @pytest.mark.my_marker
    def test_custom_composite_envelope_vector(self) -> None:
        cs1 = CustomState(3)
        cs2 = CustomState(2)
        ce = CompositeEnvelope(cs1, cs2)
        ce.combine(cs1, cs2)
        operator = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        op = Operation(CustomStateOperationType.Custom, operator=operator)
        cs1.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                jnp.array([[0], [0], [1], [0], [0], [0]]), ce.product_states[0].state
            )
        )

    def test_custom_composite_envelope_vector(self) -> None:
        cs1 = CustomState(3)
        cs2 = CustomState(2)
        ce = CompositeEnvelope(cs1, cs2)
        ce.combine(cs1, cs2)
        cs1.expand()
        cs1.expand()
        operator = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        op = Operation(CustomStateOperationType.Custom, operator=operator)
        cs1.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                jnp.array([[0], [0], [1], [0], [0], [0]]), ce.product_states[0].state
            )
        )


class TestCustomExpressionOperator(unittest.TestCase):
    def test_expression_custom_vector(self) -> None:
        cs = CustomState(3)
        context = {
            "a": lambda dims: jnp.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
            "b": lambda dims: jnp.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        }
        expr = ("add", "a", "b")
        op = Operation(CustomStateOperationType.Expresion, expr=expr, context=context)
        cs.apply_operation(op)
        self.assertEqual(cs.state, 1)

    def test_expression_custom_matrix(self) -> None:
        cs = CustomState(3)
        cs.expand()
        cs.expand()
        context = {
            "a": lambda dims: jnp.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
            "b": lambda dims: jnp.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        }
        expr = ("add", "a", "b")
        op = Operation(CustomStateOperationType.Expresion, expr=expr, context=context)
        cs.apply_operation(op)
        self.assertEqual(cs.state, 1)

    def test_expression_composite_envelope_vector(self) -> None:
        cs1 = CustomState(3)
        cs2 = CustomState(2)
        ce = CompositeEnvelope(cs1, cs2)
        ce.combine(cs1, cs2)
        context = {
            "a": lambda dims: jnp.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
            "b": lambda dims: jnp.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        }
        expr = ("add", "a", "b")
        op = Operation(CustomStateOperationType.Expresion, expr=expr, context=context)
        cs1.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                jnp.array([[0], [0], [1], [0], [0], [0]]), ce.product_states[0].state
            )
        )

    @pytest.mark.my_marker
    def test_expression_composite_envelope_matrix(self) -> None:
        cs1 = CustomState(3)
        cs2 = CustomState(2)
        ce = CompositeEnvelope(cs1, cs2)
        ce.combine(cs1, cs2)
        cs1.expand()
        context = {
            "a": lambda dims: jnp.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
            "b": lambda dims: jnp.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
        }
        expr = ("add", "a", "b")
        op = Operation(CustomStateOperationType.Expresion, expr=expr, context=context)
        cs1.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                jnp.array([[0], [0], [1], [0], [0], [0]]), ce.product_states[0].state
            )
        )
