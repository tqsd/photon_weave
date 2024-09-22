import unittest
import jax.numpy as jnp
import pytest

from photon_weave.photon_weave import Config
from photon_weave.state.fock import Fock
from photon_weave.operation import Operation, FockOperationType
from photon_weave.state.envelope import Envelope
from photon_weave.state.composite_envelope import CompositeEnvelope


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
            self.assertEqual(f.state, i+1)

    def test_creation_operation_vector(self) -> None:
        C = Config()
        C.set_contraction(False)

        f = Fock()
        f.expand()
        op = Operation(FockOperationType.Creation)
        for i in range(10):
            f.apply_operation(op)
            expected_state = jnp.zeros((i+2,1))
            expected_state = expected_state.at[i+1,0].set(1)
            self.assertTrue(
                jnp.allclose(
                    f.state,
                    expected_state
                )
            )

        C.set_contraction(True)

        f = Fock()
        f.expand()

        for i in range(10):
            f.apply_operation(op)
            expected_state = jnp.zeros((i+2,1))
            expected_state = expected_state.at[i+1,0].set(1)
            self.assertEqual(f.state, i+1)

    def test_creation_operation_matrix(self) -> None:
        C = Config()
        C.set_contraction(False)

        f = Fock()
        f.expand()
        f.expand()
        op = Operation(FockOperationType.Creation)
        for i in range(10):
            f.apply_operation(op)
            expected_state = jnp.zeros((i+2,i+2))
            expected_state = expected_state.at[i+1,i+1].set(1)
            self.assertTrue(
                jnp.allclose(
                    f.state,
                    expected_state
                )
            )

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
            expected_state = jnp.zeros(((i+2)*2,1))
            expected_state = expected_state.at[(i+1)*2,0].set(1)
            self.assertTrue(
                jnp.allclose(
                    env.state,
                    expected_state
                )
            )

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
            expected_state = jnp.zeros(((i+2)*2,1))
            expected_state = expected_state.at[(i+1)*2,0].set(1)
            self.assertTrue(
                jnp.allclose(
                    env.state,
                    expected_state
                )
            )
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
            expected_state = jnp.zeros(((i+2)*2*2,1))
            expected_state = expected_state.at[(i+1)*4,0].set(1)
            self.assertTrue(
                jnp.allclose(
                    ce.product_states[0].state,
                    expected_state
                )
            )

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
                with self.assertRaises(ValueError) as context:
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
                with self.assertRaises(ValueError) as context:
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
                with self.assertRaises(ValueError) as context:
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
                expected_state = jnp.zeros((2*env.fock.dimensions,1))
                expected_state = expected_state.at[s*2,0].set(1)
                self.assertTrue(jnp.allclose(env.state, expected_state))
            else:
                with self.assertRaises(ValueError) as context:
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
                expected_state = jnp.zeros((2*env.fock.dimensions,1))
                expected_state = expected_state.at[s*2,0].set(1)
                self.assertTrue(jnp.allclose(env.state, expected_state))
                env.expand()
            else:
                with self.assertRaises(ValueError) as context:
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
                expected_state = jnp.zeros((env.fock.dimensions,1))
                expected_state = expected_state.at[s,0].set(1)
                self.assertTrue(jnp.allclose(env.fock.trace_out(),expected_state))
            else:
                with self.assertRaises(ValueError) as context:
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
                expected_state = jnp.zeros((env.fock.dimensions,1))
                expected_state = expected_state.at[s,0].set(1)
                self.assertTrue(jnp.allclose(env.fock.trace_out(),expected_state))
            else:
                with self.assertRaises(ValueError) as context:
                    env.fock.apply_operation(op)

class TestFockOperationPhaseShift(unittest.TestCase):

    def test_phase_shift_in_place(self) -> None:
        f = Fock()
        f.state = 3
        op = Operation(FockOperationType.PhaseShift, phi=jnp.pi/2)
        f.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                f.state,
                jnp.array(
                    [[0],[0],[0],[-1j]]
                )
            )
        )

    def test_phase_shift_in_envelope_vector(self) -> None:
        env = Envelope()
        env.fock.state = 3
        env.combine()
        op = Operation(FockOperationType.PhaseShift, phi=jnp.pi/2)
        env.fock.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[0],[0],[0],[0],[0],[0],[-1j],[0]]
                )
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
        op = Operation(FockOperationType.PhaseShift, phi=jnp.pi/2)
        env.fock.apply_operation(op)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[0,0,0,0],
                     [0,0,0,0],
                     [0,0,1,0],
                     [0,0,0,0]]
                )
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

        op = Operation(FockOperationType.PhaseShift, phi=jnp.pi/2)
        #env1.fock.apply_operation(op)
        ce.apply_operation(op, env1.fock)
        
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,0]]
                )
            )
        )


class TestFockOperationDisplace(unittest.TestCase):

    @pytest.mark.my_marker
    def test_displace_fock(self) -> None:
        f = Fock()
        f.state = 0
        op = Operation(FockOperationType.Displace, alpha=4)
        f.apply_operation(op)
