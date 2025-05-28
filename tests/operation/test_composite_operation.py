import unittest

import jax.numpy as jnp
import pytest

from photon_weave._math.ops import number_operator
from photon_weave.operation import CompositeOperationType, Operation
from photon_weave.photon_weave import Config
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization, PolarizationLabel


class TestNonPolarizingBeamSplitter(unittest.TestCase):
    def test_non_polarizing_bs_vector(self) -> None:
        env1 = Envelope()
        env1.fock.state = 1
        env2 = Envelope()
        env2.fock.state = 1
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock)
        op = Operation(
            CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4
        )
        expected_state = jnp.array(
            [
                [0],
                [0],
                [1j / jnp.sqrt(2)],
                [0],
                [0],
                [0],
                [1j / jnp.sqrt(2)],
                [0],
                [0],
            ]
        )
        ce.apply_operation(op, env1.fock, env2.fock)
        self.assertTrue(
            jnp.allclose(ce.product_states[0].state, expected_state),
            ce.product_states[0].state,
        )

    def test_non_polarizing_bs_matrix(self) -> None:
        C = Config()
        C.set_contraction(True)
        env1 = Envelope()
        env1.fock.state = 1
        env2 = Envelope()
        env2.fock.state = 1
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock)
        ce.expand(env1.fock)
        op = Operation(
            CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4
        )
        ce.apply_operation(op, env1.fock, env2.fock)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [
                        [0],
                        [0],
                        [1 / jnp.sqrt(2)],
                        [0],
                        [0],
                        [0],
                        [1 / jnp.sqrt(2)],
                        [0],
                        [0],
                    ]
                ),
            )
        )


class TestExpressionOperator(unittest.TestCase):
    def test_expression_operator_on_vector(self) -> None:
        env1 = Envelope()
        env1.fock.state = 1
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.polarization)
        context = {"n": lambda dims: number_operator(dims[0])}
        op = Operation(
            CompositeOperationType.Expression,
            expr=("expm", ("s_mult", 1j, jnp.pi, "n")),
            context=context,
            state_types=(Fock,),
        )
        ce.apply_operation(op, env1.fock)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state, jnp.array([[0], [0], [-1], [0]])
            )
        )


class TestCNOTOperator(unittest.TestCase):
    def test_cnot_vector(self) -> None:
        env1 = Envelope()
        env1.polarization.state = PolarizationLabel.V
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        op = Operation(CompositeOperationType.CXPolarization)
        ce.apply_operation(op, env1.polarization, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                jnp.array([[0], [0], [0], [1]]), ce.product_states[0].state
            )
        )

    def test_cnot_matirx(self) -> None:
        env1 = Envelope()
        env1.polarization.state = PolarizationLabel.V
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        env1.polarization.expand()
        op = Operation(CompositeOperationType.CXPolarization)
        ce.apply_operation(op, env1.polarization, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                jnp.array([[0], [0], [0], [1]]), ce.product_states[0].state
            )
        )


class TestCZOperator(unittest.TestCase):
    def test_cz_operator_vector(self) -> None:
        env1 = Envelope()
        env1.polarization.state = PolarizationLabel.V
        env2 = Envelope()
        env2.polarization.state = PolarizationLabel.V
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        op = Operation(CompositeOperationType.CZPolarization)
        ce.apply_operation(op, env1.polarization, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                jnp.array([[0], [0], [0], [-1]]), ce.product_states[0].state
            )
        )

    def test_cz_operator_matrix(self) -> None:
        env1 = Envelope()
        env1.polarization.state = PolarizationLabel.V
        env2 = Envelope()
        env2.polarization.state = PolarizationLabel.V
        ce = CompositeEnvelope(env1, env2)
        env1.polarization.expand()
        ce.combine(env1.polarization, env2.polarization)
        op = Operation(CompositeOperationType.CZPolarization)
        ce.apply_operation(op, env1.polarization, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                jnp.array([[0], [0], [0], [-1]]), ce.product_states[0].state
            )
        )


class TestSWAPOperator(unittest.TestCase):
    def test_swap_operator_vector(self) -> None:
        env1 = Envelope()
        env1.polarization.state = PolarizationLabel.V
        env2 = Envelope()

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)

        op = Operation(CompositeOperationType.SwapPolarization)
        ce.apply_operation(op, env1.polarization, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                env1.polarization.trace_out(), jnp.array([[1, 0], [0, 0]])
            )
        )

        self.assertTrue(
            jnp.allclose(
                env2.polarization.trace_out(), jnp.array([[0, 0], [0, 1]])
            )
        )

    def test_swap_operator_matrix(self) -> None:
        env1 = Envelope()
        env1.polarization.state = PolarizationLabel.V
        env2 = Envelope()

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        ce.expand(env1.polarization)

        op = Operation(CompositeOperationType.SwapPolarization)
        ce.apply_operation(op, env1.polarization, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                env1.polarization.trace_out(), jnp.array([[1, 0], [0, 0]])
            )
        )

        self.assertTrue(
            jnp.allclose(
                env2.polarization.trace_out(), jnp.array([[0, 0], [0, 1]])
            )
        )


class TestCSWAPOperator(unittest.TestCase):
    def test_CSWAP_vector(self) -> None:
        env1 = Envelope()
        env1.polarization.state = PolarizationLabel.V
        env2 = Envelope()
        env2.polarization.state = PolarizationLabel.V
        env3 = Envelope()

        ce = CompositeEnvelope(env1, env2, env3)
        ce.combine(env1.polarization, env2.polarization, env3.polarization)

        op = Operation(CompositeOperationType.CSwapPolarization)
        ce.apply_operation(
            op, env1.polarization, env2.polarization, env3.polarization
        )

        self.assertTrue(
            jnp.allclose(
                env1.polarization.trace_out(), jnp.array([[0, 0], [0, 1]])
            )
        )
        self.assertTrue(
            jnp.allclose(
                env2.polarization.trace_out(), jnp.array([[1, 0], [0, 0]])
            )
        )
        self.assertTrue(
            jnp.allclose(
                env1.polarization.trace_out(), jnp.array([[0, 0], [0, 1]])
            )
        )

    def test_CSWAP_vector(self) -> None:
        env1 = Envelope()
        env1.polarization.state = PolarizationLabel.V
        env2 = Envelope()
        env2.polarization.state = PolarizationLabel.V
        env3 = Envelope()

        ce = CompositeEnvelope(env1, env2, env3)
        ce.combine(env1.polarization, env2.polarization, env3.polarization)
        ce.expand(env1.polarization)

        op = Operation(CompositeOperationType.CSwapPolarization)
        ce.apply_operation(
            op, env1.polarization, env2.polarization, env3.polarization
        )

        self.assertTrue(
            jnp.allclose(
                env1.polarization.trace_out(), jnp.array([[0, 0], [0, 1]])
            )
        )
        self.assertTrue(
            jnp.allclose(
                env2.polarization.trace_out(), jnp.array([[1, 0], [0, 0]])
            )
        )
        self.assertTrue(
            jnp.allclose(
                env1.polarization.trace_out(), jnp.array([[0, 0], [0, 1]])
            )
        )
