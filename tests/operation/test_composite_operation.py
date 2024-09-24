import pytest
import unittest
import jax.numpy as jnp

from photon_weave.operation import Operation, CompositeOperationType
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.polarization import Polarization
from photon_weave.state.fock import Fock
from photon_weave.photon_weave import Config


class TestNonPolarizingBeamSplitter(unittest.TestCase):

    @pytest.mark.my_marker
    def test_non_polarizing_bs_vector(self) -> None:
        env1 = Envelope()
        env1.fock.state = 1
        env2 = Envelope()
        env2.fock.state = 1
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)
        op = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi/4)
        ce.apply_operation(op, env1.fock, env2.fock)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0],
                    [0],
                    [1j/jnp.sqrt(2)],
                    [0],
                    [0],
                    [0],
                    [1j/jnp.sqrt(2)],
                    [0],
                    [0]]
                )
            )
        )

    def test_non_polarizing_bs_vector(self) -> None:
        env1 = Envelope()
        env1.fock.state = 1
        env2 = Envelope()
        env2.fock.state = 1
        env3 = Envelope()
        env3.fock.state = 2
        ce = CompositeEnvelope(env1,env2, env3)
        ce.combine(env3.fock, env1.fock, env2.fock)
        op = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi/4)
        ce.apply_operation(op, env1.fock, env2.fock)
        self.assertTrue(
            jnp.allclose(
                ce.trace_out(env1.fock, env2.fock),
                jnp.array(
                    [[0],
                    [0],
                    [1j/jnp.sqrt(2)],
                    [0],
                    [0],
                    [0],
                    [1j/jnp.sqrt(2)],
                    [0],
                    [0]]
                )
            )
        )

        

    @pytest.mark.my_marker
    def test_non_polarizing_bs_matrix(self) -> None:
        C = Config()
        C.set_contraction(True)
        env1 = Envelope()
        env1.fock.state = 1
        env2 = Envelope()
        env2.fock.state = 1
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)
        ce.expand(env1.fock)
        op = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi/4)
        ce.apply_operation(op, env1.fock, env2.fock)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0],
                    [0],
                    [1/jnp.sqrt(2)],
                    [0],
                    [0],
                    [0],
                    [1/jnp.sqrt(2)],
                    [0],
                    [0]]
                )
            )
        )
