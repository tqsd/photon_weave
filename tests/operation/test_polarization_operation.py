import random
import unittest

import jax.numpy as jnp
import pytest

from photon_weave.operation import Operation, PolarizationOperationType
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.polarization import Polarization, PolarizationLabel


class TestIdentityOperator(unittest.TestCase):
    def test_identity_operation_vector(self) -> None:
        p = Polarization()
        op = Operation(PolarizationOperationType.I)
        p.apply_operation(op)
        self.assertEqual(p.state, PolarizationLabel.H)
        p.state = PolarizationLabel.R
        p.apply_operation(op)
        self.assertEqual(p.state, PolarizationLabel.R)

    def test_identity_operation_matrix(self) -> None:
        p = Polarization()
        op = Operation(PolarizationOperationType.I)
        p.expand()
        p.expand()
        p.apply_operation(op)
        self.assertEqual(p.state, PolarizationLabel.H)
        p.state = PolarizationLabel.R
        p.expand()
        p.expand()
        p.apply_operation(op)
        self.assertEqual(p.state, PolarizationLabel.R)

    def test_identity_operation_envelope_vector(self) -> None:
        env = Envelope()
        op = Operation(PolarizationOperationType.I)
        env.combine()
        env.apply_operation(op, env.polarization)
        self.assertTrue(
            jnp.allclose(jnp.array([[1], [0], [0], [0], [0], [0]]), env.state)
        )

        env = Envelope()
        env.polarization.state = PolarizationLabel.R
        env.combine()
        env.apply_operation(op, env.polarization)
        self.assertTrue(
            jnp.allclose(
                jnp.array([[1 / jnp.sqrt(2)], [0], [0], [1j / jnp.sqrt(2)], [0], [0]]),
                env.state,
            )
        )

    def test_identity_operation_envelope_vector(self) -> None:
        env = Envelope()
        op = Operation(PolarizationOperationType.I)
        env.combine()
        env.expand()
        env.apply_operation(op, env.polarization)
        self.assertTrue(
            jnp.allclose(jnp.array([[1], [0], [0], [0], [0], [0]]), env.state)
        )

        env = Envelope()
        env.polarization.state = PolarizationLabel.R
        env.combine()
        env.apply_operation(op, env.polarization)
        env.reorder(env.polarization, env.fock)
        self.assertTrue(
            jnp.allclose(
                jnp.array([[1 / jnp.sqrt(2)], [0], [0], [1j / jnp.sqrt(2)], [0], [0]]),
                env.state,
            )
        )

    def test_identity_operation_composite_envelope_vector(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        op = Operation(PolarizationOperationType.I)
        ce.apply_operation(op, env2.polarization)
        self.assertTrue(
            jnp.allclose(ce.product_states[0].state, jnp.array([[1], [0], [0], [0]]))
        )

        env1 = Envelope()
        env2 = Envelope()
        env2.polarization.state = PolarizationLabel.R
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        op = Operation(PolarizationOperationType.I)
        ce.apply_operation(op, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array([[1 / jnp.sqrt(2)], [1j / jnp.sqrt(2)], [0], [0]]),
            )
        )

    def test_identity_operation_composite_envelope_matrix(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        env1.expand()
        op = Operation(PolarizationOperationType.I)
        ce.apply_operation(op, env2.polarization)
        self.assertTrue(
            jnp.allclose(ce.product_states[0].state, jnp.array([[1], [0], [0], [0]]))
        )

        env1 = Envelope()
        env2 = Envelope()
        env2.polarization.state = PolarizationLabel.R
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        env1.expand()
        op = Operation(PolarizationOperationType.I)
        ce.apply_operation(op, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array([[1 / jnp.sqrt(2)], [1j / jnp.sqrt(2)], [0], [0]]),
            )
        )


class TestXOperator(unittest.TestCase):
    def test_X_polarization_vector(self) -> None:
        p = Polarization()
        op = Operation(PolarizationOperationType.X)
        p.apply_operation(op)
        self.assertEqual(p.state, PolarizationLabel.V)

        p = Polarization()
        p.state = PolarizationLabel.V
        op = Operation(PolarizationOperationType.X)
        p.apply_operation(op)
        self.assertEqual(p.state, PolarizationLabel.H)

    def test_X_polarization_matrix(self) -> None:
        p = Polarization()
        p.expand()
        p.expand()
        op = Operation(PolarizationOperationType.X)
        p.apply_operation(op)
        self.assertEqual(p.state, PolarizationLabel.V)

        p = Polarization()
        p.state = PolarizationLabel.V
        p.expand()
        p.expand()
        op = Operation(PolarizationOperationType.X)
        p.apply_operation(op)
        self.assertEqual(p.state, PolarizationLabel.H)

    def test_X_envelope_vector(self) -> None:
        env = Envelope()
        env.combine()
        op = Operation(PolarizationOperationType.X)
        env.polarization.apply_operation(op)
        env.reorder(env.polarization, env.fock)
        self.assertTrue(
            jnp.allclose(env.state, jnp.array([[0], [0], [0], [1], [0], [0]]))
        )

        env = Envelope()
        env.polarization.state = PolarizationLabel.V
        env.combine()
        op = Operation(PolarizationOperationType.X)
        env.polarization.apply_operation(op)
        self.assertTrue(
            jnp.allclose(env.state, jnp.array([[1], [0], [0], [0], [0], [0]]))
        )

    def test_X_envelope_matrix(self) -> None:
        env = Envelope()
        env.combine()
        env.expand()
        op = Operation(PolarizationOperationType.X)
        env.polarization.apply_operation(op)
        env.reorder(env.polarization, env.fock)
        self.assertTrue(
            jnp.allclose(env.state, jnp.array([[0], [0], [0], [1], [0], [0]]))
        )

        env = Envelope()
        env.polarization.state = PolarizationLabel.V
        env.combine()
        env.expand()
        op = Operation(PolarizationOperationType.X)
        env.polarization.apply_operation(op)
        self.assertTrue(
            jnp.allclose(env.state, jnp.array([[1], [0], [0], [0], [0], [0]]))
        )

    def test_X_composite_envelope_vector(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)

        ce.combine(env1.polarization, env2.polarization)
        op = Operation(PolarizationOperationType.X)
        env1.polarization.apply_operation(op)
        self.assertTrue(
            jnp.allclose(ce.product_states[0].state, jnp.array([[0], [0], [1], [0]]))
        )

        env1 = Envelope()
        env1.polarization.state = PolarizationLabel.V
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)

        ce.combine(env1.polarization, env2.polarization)
        op = Operation(PolarizationOperationType.X)
        env1.polarization.apply_operation(op)
        self.assertTrue(
            jnp.allclose(ce.product_states[0].state, jnp.array([[1], [0], [0], [0]]))
        )

    def test_X_composite_envelope_matrix(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        env1.expand()
        op = Operation(PolarizationOperationType.X)
        env1.polarization.apply_operation(op)
        self.assertTrue(
            jnp.allclose(ce.product_states[0].state, jnp.array([[0], [0], [1], [0]]))
        )

        env1 = Envelope()
        env1.polarization.state = PolarizationLabel.V
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)

        ce.combine(env1.polarization, env2.polarization)
        env1.expand()
        op = Operation(PolarizationOperationType.X)
        env1.polarization.apply_operation(op)
        self.assertTrue(
            jnp.allclose(ce.product_states[0].state, jnp.array([[1], [0], [0], [0]]))
        )


class TestOtherOperators(unittest.TestCase):
    """
    We test if the correct operators
    are computed
    """

    def test_H_operation(self) -> None:
        op = Operation(PolarizationOperationType.H)
        op.compute_dimensions(0, 0)
        H = jnp.array([[1, 1], [1, -1]])
        self.assertTrue(jnp.allclose(1 / jnp.sqrt(2) * H, op.operator))

    def test_S_operator(self) -> None:
        op = Operation(PolarizationOperationType.S)
        op.compute_dimensions(0, 0)
        S = jnp.array([[1, 0], [0, 1j]])
        self.assertTrue(jnp.allclose(op.operator, S))

    def test_T_operator(self) -> None:
        op = Operation(PolarizationOperationType.T)
        op.compute_dimensions(0, 0)
        T = jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]])
        self.assertTrue(jnp.allclose(op.operator, T))

    def test_SX_operator(self) -> None:
        op = Operation(PolarizationOperationType.SX)
        op.compute_dimensions(0, 0)
        SX = 1 / 2 * jnp.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
        self.assertTrue(jnp.allclose(op.operator, SX))

    def test_RX_operator(self) -> None:
        theta = random.uniform(0, 2 * jnp.pi)
        op = Operation(PolarizationOperationType.RX, theta=theta)
        op.compute_dimensions(0, 0)
        RX = jnp.array(
            [
                [jnp.cos(theta / 2), -1j * jnp.sin(theta / 2)],
                [-1j * jnp.sin(theta / 2), jnp.cos(theta / 2)],
            ]
        )
        self.assertTrue(jnp.allclose(op.operator, RX))

    def test_RY_operator(self) -> None:
        theta = random.uniform(0, 2 * jnp.pi)
        op = Operation(PolarizationOperationType.RY, theta=theta)
        op.compute_dimensions(0, 0)
        RY = jnp.array(
            [
                [jnp.cos(theta / 2), -jnp.sin(theta / 2)],
                [jnp.sin(theta / 2), jnp.cos(theta / 2)],
            ]
        )
        self.assertTrue(jnp.allclose(op.operator, RY))

    def test_RZ_operator(self) -> None:
        theta = random.uniform(0, 2 * jnp.pi)
        op = Operation(PolarizationOperationType.RZ, theta=theta)
        op.compute_dimensions(0, 0)
        RZ = jnp.array([[jnp.exp(-1j * theta / 2), 0], [0, jnp.exp(1j * theta / 2)]])
        self.assertTrue(jnp.allclose(op.operator, RZ))

    def test_U3_operator(self) -> None:
        phi = random.uniform(0, 2 * jnp.pi)
        theta = random.uniform(0, 2 * jnp.pi)
        omega = random.uniform(0, 2 * jnp.pi)
        op = Operation(PolarizationOperationType.U3, phi=phi, theta=theta, omega=omega)
        op.compute_dimensions(0, 0)
        cos_term = jnp.cos(theta / 2)
        sin_term = jnp.sin(theta / 2)

        U3 = jnp.array(
            [
                [cos_term, -jnp.exp(1j * omega) * sin_term],
                [jnp.exp(1j * phi) * sin_term, jnp.exp(1j * (phi + omega)) * cos_term],
            ]
        )
        self.assertTrue(jnp.allclose(op.operator, U3))

    def test_custom_operator(self) -> None:
        matrix = [[random.uniform(0, 1) for _ in range(2)] for _ in range(2)]
        operator = jnp.array(matrix)
        op = Operation(PolarizationOperationType.Custom, operator=operator)
        op.compute_dimensions(0, 0)
        self.assertTrue(jnp.allclose(op.operator, operator))
