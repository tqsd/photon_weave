import unittest

import jax.numpy as jnp
import pytest

from photon_weave._math.ops import (
    annihilation_operator,
    creation_operator,
    number_operator,
)
from photon_weave.operation import FockOperationType, Operation
from photon_weave.photon_weave import Config
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock


class TestFockOperationIdentity(unittest.TestCase):
    def test_identity_operation_vector(self) -> None:
        fo = Fock()
        fo.state = 2
        fo.dimensions = 3
        op = Operation(FockOperationType.Identity)
        fo.apply_operation(op)
        self.assertEqual(fo.state, 2)


class TestFockOperationCreation(unittest.TestCase):
    def test_creation_operation_label(self) -> None:
        C = Config()
        C.set_contraction(True)
        f = Fock()
        op = Operation(FockOperationType.Creation)
        for i in range(20):
            f.apply_operation(op)
            self.assertEqual(f.state, i + 1)

    def test_creation_operation_vector(self) -> None:
        C = Config()
        C.set_contraction(False)

        f = Fock()
        f.expand()
        op = Operation(FockOperationType.Creation)
        for i in range(10):
            f.apply_operation(op)
            expected_state = jnp.zeros((i + 2, 1))
            expected_state = expected_state.at[i + 1, 0].set(1)
            self.assertTrue(jnp.allclose(f.state, expected_state))

        C.set_contraction(True)

        f = Fock()
        f.expand()

        for i in range(10):
            f.apply_operation(op)
            expected_state = jnp.zeros((i + 2, 1))
            expected_state = expected_state.at[i + 1, 0].set(1)
            self.assertEqual(f.state, i + 1)

    def test_creation_operation_matrix(self) -> None:
        C = Config()
        C.set_contraction(False)

        f = Fock()
        f.expand()
        f.expand()
        op = Operation(FockOperationType.Creation)
        for i in range(10):
            f.apply_operation(op)
            expected_state = jnp.zeros((i + 2, i + 2))
            expected_state = expected_state.at[i + 1, i + 1].set(1)
            self.assertTrue(jnp.allclose(f.state, expected_state))

    def test_creation_operation_envelope_vector(self) -> None:
        C = Config()
        C.set_contraction(False)
        env = Envelope()
        env.fock.expand()
        env.polarization.expand()
        env.combine()
        op = Operation(FockOperationType.Creation)
        for i in range(5):
            env.apply_operation(op, env.fock)
            expected_state = jnp.zeros(((i + 2) * 2, 1))
            expected_state = expected_state.at[(i + 1) * 2, 0].set(1)
            self.assertTrue(jnp.allclose(env.state, expected_state))

    def test_creation_operation_envelope_matrix(self) -> None:
        C = Config()
        C.set_contraction(True)
        env = Envelope()
        env.fock.expand()
        env.polarization.expand()
        env.combine()
        env.expand()
        op = Operation(FockOperationType.Creation)
        for i in range(5):
            env.apply_operation(op, env.fock)
            expected_state = jnp.zeros(((i + 2) * 2, 1))
            expected_state = expected_state.at[(i + 1) * 2, 0].set(1)
            self.assertTrue(jnp.allclose(env.state, expected_state))
            env.expand()

    def test_creation_operation_composite_envelope_envelope(self) -> None:
        C = Config()
        C.set_contraction(True)
        env1 = Envelope()
        env1.fock.expand()
        env1.polarization.expand()
        env1.combine()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.polarization)
        op = Operation(FockOperationType.Creation)
        for i in range(5):
            env1.apply_operation(op, env1.fock)
            expected_state = jnp.zeros(((i + 2) * 2 * 2, 1))
            expected_state = expected_state.at[(i + 1) * 4, 0].set(1)
            self.assertTrue(jnp.allclose(ce.product_states[0].state, expected_state))


class TestFockOperationAnnihilation(unittest.TestCase):
    def test_destruction_operation_label(self) -> None:
        C = Config()
        C.set_contraction(True)
        f = Fock()
        f.state = 10
        expected_state = 10
        op = Operation(FockOperationType.Annihilation)
        for _ in range(11):
            expected_state -= 1
            if expected_state >= 0:
                f.apply_operation(op)
                self.assertEqual(f.state, expected_state)
            else:
                with self.assertRaises(ValueError):
                    f.apply_operation(op)

    def test_destruction_operation_vector(self) -> None:
        C = Config()
        C.set_contraction(True)
        f = Fock()
        f.state = 10
        f.expand()
        expected_state = 10
        op = Operation(FockOperationType.Annihilation)
        for _ in range(11):
            expected_state -= 1
            if expected_state >= 0:
                f.apply_operation(op)
                self.assertEqual(f.state, expected_state)
                f.expand()
            else:
                with self.assertRaises(ValueError):
                    f.apply_operation(op)
                    f.expand()

    def test_destruction_operation_matrix(self) -> None:
        C = Config()
        C.set_contraction(True)
        f = Fock()
        f.state = 10
        f.expand()
        f.expand()
        expected_state = 10
        op = Operation(FockOperationType.Annihilation)
        for _ in range(11):
            expected_state -= 1
            if expected_state >= 0:
                f.apply_operation(op)
                self.assertEqual(f.state, expected_state)
                f.expand()
                f.expand()
            else:
                with self.assertRaises(ValueError):
                    f.apply_operation(op)
                    f.expand()
                    f.expand()

    def test_destruction_operation_vector_envelope(self) -> None:
        C = Config()
        C.set_contraction(True)
        env = Envelope()
        env.fock.state = 2
        env.combine()
        s = 2
        op = Operation(FockOperationType.Annihilation)

        for _ in range(5):
            s -= 1
            if s >= 0:
                env.fock.apply_operation(op)
                expected_state = jnp.zeros((2 * env.fock.dimensions, 1))
                expected_state = expected_state.at[s * 2, 0].set(1)
                self.assertTrue(jnp.allclose(env.state, expected_state))
            else:
                with self.assertRaises(ValueError):
                    env.fock.apply_operation(op)
                    env.expand()

    def test_destruction_operation_matrix_envelope(self) -> None:
        C = Config()
        C.set_contraction(True)
        env = Envelope()
        env.fock.state = 2
        env.fock.dimensions = 3
        env.combine()
        env.expand()
        s = 2
        op = Operation(FockOperationType.Annihilation)

        for _ in range(5):
            s -= 1
            if s >= 0:
                env.fock.apply_operation(op)
                expected_state = jnp.zeros((2 * env.fock.dimensions, 1))
                expected_state = expected_state.at[s * 2, 0].set(1)
                self.assertTrue(jnp.allclose(env.state, expected_state))
                env.expand()
            else:
                with self.assertRaises(ValueError):
                    env.fock.apply_operation(op)
                    env.expand()

    def test_destruction_operation_vector_composite_envelope(self) -> None:
        env = Envelope()
        oenv = Envelope()
        env.fock.state = 3
        ce = CompositeEnvelope(env, oenv)
        ce.combine(oenv.fock, env.fock)
        op = Operation(FockOperationType.Annihilation)
        s = 3
        for _ in range(5):
            s -= 1
            if s >= 0:
                env.fock.apply_operation(op)
                expected_state = jnp.zeros((env.fock.dimensions, 1))
                expected_state = expected_state.at[s, 0].set(1)
                self.assertTrue(jnp.allclose(env.fock.trace_out(), expected_state))
            else:
                with self.assertRaises(ValueError):
                    env.fock.apply_operation(op)

    def test_destruction_operation_matrix_composite_envelope(self) -> None:
        env = Envelope()
        oenv = Envelope()
        env.fock.state = 3
        env.fock.uid = "f"
        ce = CompositeEnvelope(env, oenv)
        oenv.polarization.expand()
        oenv.polarization.uid = "p"
        env.fock.expand()
        ce.combine(oenv.polarization, env.fock)
        env.expand()
        op = Operation(FockOperationType.Annihilation)
        s = 3
        for _ in range(5):
            s -= 1
            if s >= 0:
                env.fock.expand()
                env.fock.apply_operation(op)
                expected_state = jnp.zeros((env.fock.dimensions, 1))
                expected_state = expected_state.at[s, 0].set(1)
                self.assertTrue(jnp.allclose(env.fock.trace_out(), expected_state))
            else:
                with self.assertRaises(ValueError):
                    env.fock.apply_operation(op)


class TestFockOperationPhaseShift(unittest.TestCase):
    def test_phase_shift_in_place(self) -> None:
        f = Fock()
        f.state = 3
        op = Operation(FockOperationType.PhaseShift, phi=jnp.pi / 2)
        f.apply_operation(op)
        self.assertTrue(jnp.allclose(f.state, jnp.array([[0], [0], [0], [-1j]])))

    def test_phase_shift_in_envelope_vector(self) -> None:
        env = Envelope()
        env.fock.state = 3
        env.combine()
        op = Operation(FockOperationType.PhaseShift, phi=jnp.pi / 2)
        env.fock.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                env.state, jnp.array([[0], [0], [0], [0], [0], [0], [-1j], [0]])
            )
        )

    def test_phase_shift_in_envelope_matrix(self) -> None:
        C = Config()
        C.set_contraction(False)
        env = Envelope()
        env.fock.state = 1
        env.fock.dimensions = 2
        env.combine()
        env.expand()
        op = Operation(FockOperationType.PhaseShift, phi=jnp.pi / 2)
        env.fock.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]),
            )
        )

    def test_phase_shift_in_composite_envelope_matrix(self) -> None:
        C = Config()
        C.set_contraction(False)
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.state = 2
        env1.fock.dimensions = 4
        env2.fock.state = 1
        env2.fock.dimensions = 2

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env2.polarization, env1.fock)
        env1.expand()

        op = Operation(FockOperationType.PhaseShift, phi=jnp.pi / 2)
        # env1.fock.apply_operation(op)
        ce.apply_operation(op, env1.fock)

        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            )
        )


class TestFockOperationDisplace(unittest.TestCase):
    def test_displace_fock_vector(self) -> None:
        f = Fock()
        f.state = 0
        f.dimensions = 2
        op = Operation(FockOperationType.Displace, alpha=1)
        f.apply_operation(op)
        self.assertEqual(f.dimensions, 12)

    def test_displace_fock_matrix(self) -> None:
        f = Fock()
        f.state = 0
        f.dimensions = 2
        f.expand()
        f.expand()
        op = Operation(FockOperationType.Displace, alpha=2)
        f.apply_operation(op)
        self.assertEqual(f.dimensions, 20)

    def test_displace_fock_envelope_vector(self) -> None:
        env = Envelope()
        env.combine()
        op = Operation(FockOperationType.Displace, alpha=2)
        env.apply_operation(op, env.fock)
        self.assertEqual(env.fock.dimensions, 20)
        self.assertTrue(env.state.shape == (env.dimensions, 1))

    def test_displace_fock_envelope_matrix(self) -> None:
        C = Config()
        C.set_contraction(True)
        env = Envelope()
        env.combine()
        env.expand()
        op = Operation(FockOperationType.Displace, alpha=2)
        env.apply_operation(op, env.fock)
        self.assertEqual(env.fock.dimensions, 20)
        self.assertTrue(env.state.shape == (env.dimensions, 1))

    def test_displace_fock_composite_envelope_vector(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock)
        op = Operation(FockOperationType.Displace, alpha=2)
        env1.fock.apply_operation(op)
        self.assertTrue(
            ce.product_states[0].state.shape
            == (env1.fock.dimensions * env2.fock.dimensions, 1)
        )

    def test_displace_fock_composite_envelope_matrix(self) -> None:
        C = Config()
        C.set_contraction(True)
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock)
        ce.expand(env1.fock)
        op = Operation(FockOperationType.Displace, alpha=2)
        env1.fock.apply_operation(op)
        self.assertTrue(
            ce.product_states[0].state.shape
            == (env1.fock.dimensions * env2.fock.dimensions, 1)
        )


class TestFockOperationSqueeze(unittest.TestCase):
    def test_squeeze_fock_vector(self) -> None:
        C = Config()
        C.set_contraction(True)
        f = Fock()
        op = Operation(FockOperationType.Squeeze, zeta=0.5)
        f.apply_operation(op)
        self.assertTrue(f.state.shape == (f.dimensions, 1))

    def test_squeeze_envelope_vector(self) -> None:
        C = Config()
        C.set_contraction(True)
        env = Envelope()
        env.combine()
        op = Operation(FockOperationType.Squeeze, zeta=0.5j)
        env.apply_operation(op, env.fock)
        self.assertTrue(env.state.shape == (env.dimensions, 1))

    def test_squeeze_envelope_matrix(self) -> None:
        C = Config()
        C.set_contraction(True)
        env = Envelope()
        env.combine()
        env.expand()
        op = Operation(FockOperationType.Squeeze, zeta=0.5j)
        env.apply_operation(op, env.fock)
        self.assertTrue(env.state.shape == (env.dimensions, 1))

    def test_squeeze_composite_envelope_vector(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock)
        op = Operation(FockOperationType.Squeeze, zeta=-0.5j)
        env1.fock.apply_operation(op)
        self.assertTrue(
            ce.product_states[0].state.shape
            == (env1.fock.dimensions * env2.fock.dimensions, 1)
        )


class TestExpressionOperator(unittest.TestCase):
    def test_expression_operator_fock_vector(self) -> None:
        f = Fock()
        f.state = 1
        context = {
            "a": lambda dims: annihilation_operator(dims[0]),
            "a_dag": lambda dims: creation_operator(dims[0]),
            "n": lambda dims: number_operator(dims[0]),
        }
        op = Operation(
            FockOperationType.Expresion,
            expr=("expm", ("s_mult", -1j, jnp.pi, "n")),
            context=context,
        )
        f.apply_operation(op)
        self.assertTrue(jnp.allclose(f.state, jnp.array([[0], [-1], [0], [0]])))

    def test_expression_operator_fock_vector_two_scalers(self) -> None:
        f = Fock()
        # DISPLACE
        context = {
            "a": lambda dims: annihilation_operator(dims[0]),
            "a_dag": lambda dims: creation_operator(dims[0]),
            "n": lambda dims: number_operator(dims[0]),
        }
        op = Operation(
            FockOperationType.Expresion,
            expr=("expm", ("sub", ("s_mult", 1, "a_dag"), ("s_mult", 1, "a"))),
            context=context,
        )
        f.apply_operation(op)
        self.assertTrue(f.state.shape == (12, 1))

    def test_expression_operator_fock_matrix(self) -> None:
        f = Fock()
        context = {
            "a": lambda dims: annihilation_operator(dims[0]),
            "a_dag": lambda dims: creation_operator(dims[0]),
            "n": lambda dims: number_operator(dims[0]),
        }
        op = Operation(
            FockOperationType.Expresion,
            expr=("expm", ("s_mult", -1j, jnp.pi, "n")),
            context=context,
        )
        f.expand()
        f.expand()
        f.apply_operation(op)
        self.assertEqual(f.state, 0)

    def test_expression_operator_envelope_vector(self) -> None:
        env = Envelope()
        env.fock.state = 1
        context = {
            "a": lambda dims: annihilation_operator(dims[0]),
            "a_dag": lambda dims: creation_operator(dims[0]),
            "n": lambda dims: number_operator(dims[0]),
        }
        op = Operation(
            FockOperationType.Expresion,
            expr=("expm", ("s_mult", -1j, jnp.pi, "n")),
            context=context,
        )
        env.combine()
        env.fock.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                env.state, jnp.array([[0], [0], [-1], [0], [0], [0], [0], [0]])
            )
        )

    def test_expression_operator_envelope_matrix(self) -> None:
        env = Envelope()
        env.fock.state = 1
        context = {
            "a": lambda dims: annihilation_operator(dims[0]),
            "a_dag": lambda dims: creation_operator(dims[0]),
            "n": lambda dims: number_operator(dims[0]),
        }
        op = Operation(
            FockOperationType.Expresion,
            expr=("expm", ("s_mult", -1j, jnp.pi, "n")),
            context=context,
        )
        env.combine()
        env.expand()
        env.fock.apply_operation(op)
        # Global phase is lost in a density matrix
        self.assertTrue(
            jnp.allclose(env.state, jnp.array([[0], [0], [1], [0], [0], [0], [0], [0]]))
        )

    def test_expression_operator_composite_envelope_vector(self) -> None:
        env1 = Envelope()
        env1.fock.state = 1
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock)
        context = {
            "a": lambda dims: annihilation_operator(dims[0]),
            "a_dag": lambda dims: creation_operator(dims[0]),
            "n": lambda dims: number_operator(dims[0]),
        }
        op = Operation(
            FockOperationType.Expresion,
            expr=("expm", ("s_mult", -1j, jnp.pi, "n")),
            context=context,
        )
        env1.fock.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array([[0], [-1], [0], [0]]),
            )
        )

    def test_expression_operator_composite_envelope_matrix(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock)
        env1.fock.expand()
        context = {
            "a": lambda dims: annihilation_operator(dims[0]),
            "a_dag": lambda dims: creation_operator(dims[0]),
            "n": lambda dims: number_operator(dims[0]),
        }
        op = Operation(
            FockOperationType.Expresion,
            expr=("expm", ("s_mult", -1j, jnp.pi, "n")),
            context=context,
        )
        env1.fock.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array([[1], [0], [0], [0]]),
            )
        )


class TestCustomFockOperation(unittest.TestCase):
    def test_custom_operator_fock_vector(self) -> None:
        f = Fock()
        f.dimensions = 2
        op = Operation(FockOperationType.Custom, operator=jnp.array([[0, 0], [1, 0]]))
        f.apply_operation(op)
        self.assertEqual(f.state, 1)

    def test_custom_opeator_fock_matrix(self) -> None:
        f = Fock()
        f.dimensions = 2
        f.expand()
        f.expand()
        op = Operation(FockOperationType.Custom, operator=jnp.array([[0, 0], [1, 0]]))
        f.apply_operation(op)
        self.assertEqual(f.state, 1)

    def test_custom_operator_envelope_vector(self) -> None:
        env = Envelope()
        env.fock.dimensions = 2
        env.combine()
        op = Operation(FockOperationType.Custom, operator=jnp.array([[0, 0], [1, 0]]))
        env.fock.apply_operation(op)
        self.assertTrue(jnp.allclose(env.state, jnp.array([[0], [0], [1], [0]])))

    def test_custom_operator_envelope_matrix(self) -> None:
        env = Envelope()
        env.fock.dimensions = 2
        env.combine()
        env.expand()
        op = Operation(FockOperationType.Custom, operator=jnp.array([[0, 0], [1, 0]]))
        env.fock.apply_operation(op)
        self.assertTrue(jnp.allclose(env.state, jnp.array([[0], [0], [1], [0]])))

    def test_custom_operator_composite_envelope_vector(self) -> None:
        env1 = Envelope()
        env1.fock.dimensions = 2
        env2 = Envelope()
        env2.fock.dimensions = 2
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock)
        op = Operation(FockOperationType.Custom, operator=jnp.array([[0, 0], [1, 0]]))
        env1.fock.apply_operation(op)
        env1.fock.uid = 1
        env2.fock.uid = 2
        # print([s.uid for s in ce.product_states[0].state_objs])
        self.assertTrue(
            jnp.allclose(ce.product_states[0].state, jnp.array([[0], [1], [0], [0]]))
        )

    def test_custom_operator_composite_envelope_matrix(self) -> None:
        env1 = Envelope()
        env1.fock.dimensions = 2
        env2 = Envelope()
        env2.fock.dimensions = 2
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock)
        ce.expand(env1.fock)
        op = Operation(FockOperationType.Custom, operator=jnp.array([[0, 0], [1, 0]]))
        env1.fock.apply_operation(op)
        env1.fock.uid = 1
        env2.fock.uid = 2
        # print([s.uid for s in ce.product_states[0].state_objs])
        self.assertTrue(
            jnp.allclose(ce.product_states[0].state, jnp.array([[0], [1], [0], [0]]))
        )
