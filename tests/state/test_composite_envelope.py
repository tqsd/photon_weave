import unittest
import pytest
import jax.numpy as jnp
import jax
from typing import List, Union

from photon_weave.photon_weave import Config
from photon_weave.state.composite_envelope import CompositeEnvelope, CompositeEnvelopeContainer
from photon_weave.state.envelope import Envelope
from photon_weave.state.polarization import Polarization, PolarizationLabel
from photon_weave.state.fock import Fock
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.custom_state import CustomState


class TestCompositeEnvelopeInitialization(unittest.TestCase):
    """
    Test Initialization functionality
    """
    def test_initialization_empty(self) -> None:
        """
        Test initialization of empty composite envelope
        """
        ce = CompositeEnvelope()
        self.assertIsNotNone(
            CompositeEnvelope._containers[ce.uid]
        )
        self.assertEqual(
            ce.envelopes,
            []
        )
        self.assertEqual(
            ce.states,
            []
        )

    def test_initialization_one_envelope(self) -> None:
        """
        Test the initialization with one envelope
        """
        env = Envelope()
        ce = CompositeEnvelope(env)
        self.assertEqual(ce.envelopes, [env])
        self.assertEqual(ce.states, [])

    def test_initialization_two_envelopes(self) -> None:
        """
        Test the initialization of composite envelope
        with two envelopes
        """
        env1 = Envelope()
        env2 = Envelope()
        ce = CompositeEnvelope(env1, env2)
        self.assertEqual(ce.states, [])
        for env in [env1, env2]:
            self.assertTrue(env in ce.envelopes)

    def test_initialization_two_envelopes_two_custom_states(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        cs1 = CustomState(2)
        cs2 = CustomState(2)
        ce = CompositeEnvelope(env1, env2, cs1, cs2)

    def test_initialization_three_envelopes_two_composite_envelopes(self) -> None:
        """
        Test the initialization of the composite envelopes
        using second composite envelope
        """
        env1 = Envelope()
        env2 = Envelope()
        cs = CustomState(2)
        ce1 = CompositeEnvelope(env1, env2, cs)
        self.assertEqual(ce1.states, [])
        for env in [env1, env2]:
            self.assertTrue(env in ce1.envelopes)
        env3 = Envelope()
        env4 = Envelope()
        ce2 = CompositeEnvelope(env3, env4, ce1)


        self.assertEqual(ce2.states, [])
        for env in [env1, env2, env3, env4]:
            self.assertTrue(env in ce2.envelopes)
        self.assertEqual(ce1.uid, ce2.uid)
        self.assertEqual(ce1.states, ce2.states)
        self.assertEqual(ce1.envelopes, ce2.envelopes)

    def test_initialization_envelope_from_another_composite_envelopes(self) -> None:
        """
        Test the initialization of composite envelope
        where envelope is in another composite envelope
        """
        env1 = Envelope()
        env2 = Envelope()
        ce1 = CompositeEnvelope(env1, env2)
        self.assertEqual(ce1.states, [])
        for env in [env1, env2]:
            self.assertTrue(env in ce1.envelopes)
            self.assertTrue(env.composite_envelope_id == ce1.uid)
            self.assertTrue(env.composite_envelope.uid == ce1.uid)

        env3 = Envelope()
        env3.fock.label = 1
        ce2 = CompositeEnvelope(env1, env3)
        for env in [env1, env2, env3]:
            self.assertTrue(env in ce2.envelopes)

    def test_initialization_with_combined_composite_envelope(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.fock, env2.fock)
        env3 = Envelope()
        env4 = Envelope()
        ce2 = CompositeEnvelope(env3, env4)

        ce3 = CompositeEnvelope(ce1, ce2)
        for env in [env1,env2,env3,env4]:
            self.assertTrue(env in ce1.envelopes)
            self.assertTrue(env in ce2.envelopes)

        self.assertEqual(ce1.uid, ce2.uid)

    def test_initialization_with_combined_envelope(self) -> None:
        """
        Test the case, when the combined envelope is combined with
        the composite envelope
        """
        env1 = Envelope()
        env1.combine()
        env2 = Envelope()
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)
        self.assertEqual(ce.product_states[0].state_objs, [env1.fock, env1.polarization, env2.fock])
        self.assertIsNone(env1.state)
        self.assertIsNone(env1.fock.state)
        self.assertIsNone(env1.polarization.state)


class TestStateCombining(unittest.TestCase):
    """
    Test different cases of the State joining
    """
    def test_simple_combine(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.state = 1

        ce1 = CompositeEnvelope(env1, env2)
        env1.fock.expand()
        env2.polarization.expand()
        ce1.combine(env1.fock, env2.polarization)

        self.assertEqual(env1.fock.index, (0,0))
        self.assertEqual(env2.polarization.index, (0,1))
        self.assertIsNone(env1.fock.state)
        self.assertIsNone(env2.polarization.state)
        self.assertTrue(
            jnp.allclose(
                ce1.product_states[0].state,
                jnp.array(
                    [[0],[0],[1],[0],[0],[0],[0],[0]]
                )
            )
        )

    def test_combine_no_effect(self) -> None:
        """
        Test the case where user wants
        to combined already combined states
        """
        env1 = Envelope()
        env2 = Envelope()

        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.polarization, env2.polarization)
        ce1.combine(env1.polarization, env2.polarization)

        self.assertEqual(env1.polarization.index, (0,0))
        self.assertEqual(env2.polarization.index, (0,1))
        self.assertIsNone(env1.polarization.state)
        self.assertIsNone(env2.polarization.state)
        self.assertIsNone(env1.polarization.state)
        self.assertIsNone(env2.polarization.state)
        self.assertTrue(
            jnp.allclose(
                ce1.product_states[0].state,
                jnp.array(
                    [[1],[0],[0],[0]]
                )
            )
        )

    def test_combine_with_custom_state(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        cs = CustomState(2)

        ce = CompositeEnvelope(env1, env2, cs)
        ce.combine(env1.polarization, env2.polarization, cs)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[1],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0],
                     [0]]
                )
            )
        )
        self.assertIsNone(env1.polarization.state)
        self.assertIsNone(env2.polarization.state)
        self.assertIsNone(cs.state)


    def test_combine_with_combined_composite_envelope(self) -> None:
        env1 = Envelope()
        env1.polarization.uid = "1"
        env2 = Envelope()
        env2.polarization.uid = "2"

        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.polarization, env2.polarization)
        
        env3 = Envelope()
        env3.polarization.uid = "3"
        ce2 = CompositeEnvelope(env1, env3)
        ce2.combine(env1.polarization, env3.polarization)

        self.assertEqual(env1.polarization.index, (0,0))
        self.assertEqual(env2.polarization.index, (0,1))
        self.assertIsNone(env1.polarization.state)
        self.assertIsNone(env2.polarization.state)
        self.assertTrue(
            jnp.allclose(
                ce1.product_states[0].state,
                jnp.array(
                    [[1],[0],[0],[0],[0],[0],[0],[0]]
                )
            )
        )

    def test_combine_with_combined_envelope(self) -> None:
        env1 = Envelope()
        env1.polarization.uid = "p"
        env1.fock.uid = "f"
        env1.fock.dimensions = 2
        env1.combine()

        env2 = Envelope()
        env2.polarization.uid = "2"
        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.polarization, env2.polarization)

        self.assertEqual(env1.fock.index, (0,0))
        self.assertEqual(env1.polarization.index, (0,1))
        self.assertEqual(env2.polarization.index, (0,2))

        self.assertIsNone(env1.polarization.state)
        self.assertIsNone(env2.polarization.state)
        self.assertTrue(
            jnp.allclose(
                ce1.product_states[0].state,
                jnp.array(
                    [[1],[0],[0],[0],[0],[0],[0],[0]]
                )
            )
        )

    def test_combine_matrix(self) -> None:
        env1 = Envelope()
        env1.fock.dimensions = 2
        env1.expand()
        env1.expand()
        env2 = Envelope()

        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.fock, env2.polarization)

        self.assertEqual(env1.fock.index, (0,0))
        self.assertEqual(env2.polarization.index, (0,1))
        self.assertIsNone(env1.fock.state)
        self.assertIsNone(env2.polarization.state)
        self.assertTrue(
            jnp.allclose(
                ce1.product_states[0].state,
                jnp.array(
                    [[1,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]]
                )
            )
        )

    def test_combine_expanded_state_with_envelope(self) -> None:
        env1 = Envelope()
        env1.fock.dimensions= 2
        env2 = Envelope()
        ce1 = CompositeEnvelope(env1, env2)
        ce1.combine(env1.polarization, env2.polarization)
        env1.fock.expand()
        env1.fock.expand()
        ce1.combine(env1.polarization, env1.fock)

        self.assertEqual(env1.polarization.index, (0,0))
        self.assertEqual(env2.polarization.index, (0,1))
        self.assertEqual(env1.fock.index, (0,2))
        self.assertIsNone(env1.fock.state)
        self.assertIsNone(env2.polarization.state)
        self.assertTrue(
            jnp.allclose(
                ce1.product_states[0].state,
                jnp.array(
                    [[1,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0]]
                )
            )
        )

    def test_combine_state_with_matrix_envelope(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env3 = Envelope()
        env3.fock.dimensions = 2
        env3.combine()
        env2.fock.dimensions = 2

        ce1 = CompositeEnvelope(env1, env2, env3)
        ce1.combine(env1.polarization, env2.polarization)
        ce1.product_states[0].expand()

        ce1.combine(env1.polarization, env3.fock)

        self.assertEqual(env1.polarization.index, (0,0))
        self.assertEqual(env2.polarization.index, (0,1))
        self.assertEqual(env3.fock.index, (0,2))
        self.assertEqual(env3.polarization.index, (0,3))
        self.assertEqual(
            [env1.polarization, env2.polarization, env3.fock, env3.polarization],
            ce1.product_states[0].state_objs
        )

        self.assertIsNone(env2.polarization.state)
        self.assertIsNone(env3.polarization.state)
        self.assertIsNone(env3.fock.state)

        self.assertEqual(
            ce1.product_states[0].state.shape,
            (16,16)
        )
        self.assertEqual(
            ce1.product_states[0].state[0,0],
            1
        )


class TestRepresentationMethod(unittest.TestCase):
    def test_repr(self) -> None:
        env1 = Envelope()
        env1.uid = "e1"
        env1.fock.uid= "f1"
        env1.polarization.uid= "p1"
        env2 = Envelope()
        env2.uid = "e2"
        env2.fock.uid= "f2"
        env2.polarization.uid= "p2"
        ce = CompositeEnvelope(env1, env2)
        self.assertEqual(
            f"CompositeEnvelope(uid={ce.uid}, envelopes=['e1', 'e2'], state_objects=['f1', 'p1', 'f2', 'p2'])",
            ce.__repr__()
        )

    def test_repr_with_custom_state(self) -> None:
        env1 = Envelope()
        env1.uid = "e"
        env1.fock.uid = "f"
        env1.polarization.uid = "p"
        cs = CustomState(2)
        cs.uid = "c"
        ce = CompositeEnvelope(env1, cs)
        self.assertEqual(
            ce.__repr__(),
            f"CompositeEnvelope(uid={ce.uid}, envelopes=['{env1.uid}'], state_objects=['f', 'p', 'c'])",
        )
        


class TestProductStateReordering(unittest.TestCase):
    def test_vector_reordering(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.polarization.uid = "P1"
        env2.polarization.uid = "P2"
        env2.polarization.state = PolarizationLabel.V

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array([[0],[1],[0],[0]])
            )
        )
        self.assertEqual(env1.polarization.index, (0,0))
        self.assertEqual(env2.polarization.index, (0,1))
        ce.reorder(env2.polarization, env1.polarization)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array([[0],[0],[1],[0]])
            )
        )
        self.assertEqual(env1.polarization.index, (0,1))
        self.assertEqual(env2.polarization.index, (0,0))

    def test_vector_reordering_three_states(self) -> None:
        env1 = Envelope()
        env2 = Envelope()

        env1.fock.dimensions = 2
        env2.fock.dimensions = 2

        env1.polarization.state = PolarizationLabel.R

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.polarization, env1.fock, env2.fock)
        self.assertEqual(env1.polarization.index,(0,0))
        self.assertEqual(env1.fock.index,(0,1))
        self.assertEqual(env2.fock.index,(0,2))

        first_expected_state = 1/jnp.sqrt(2) * jnp.array(
            [[1],[0],[0],[0],[1j],[0],[0],[0]]
        )
        self.assertTrue(
            jnp.allclose(
                first_expected_state,
                ce.product_states[0].state
            )
        )

        ce.reorder(env1.fock, env2.fock, env1.polarization)
        self.assertEqual(env1.fock.index,(0,0))
        self.assertEqual(env2.fock.index,(0,1))
        self.assertEqual(env1.polarization.index,(0,2))
        second_expected_state = 1/jnp.sqrt(2) * jnp.array(
            [[1],[1j],[0],[0],[0],[0],[0],[0]]
        )
        self.assertTrue(
            jnp.allclose(
                second_expected_state,
                ce.product_states[0].state
            )
        )

    def test_matrix_reordering(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env2.polarization.state = PolarizationLabel.V

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        ce.product_states[0].expand()
        ce.reorder(env2.polarization, env1.polarization)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0,0,0,0],
                     [0,0,0,0],
                     [0,0,1,0],
                     [0,0,0,0]]
                )
            )
        )
        self.assertEqual(env1.polarization.index, (0,1))
        self.assertEqual(env2.polarization.index, (0,0))


class TestCompositeEnvelopeMeasurementsVectors(unittest.TestCase):
    def test_simple_measurement_vector(self) -> None:
        C = Config()
        C.set_seed(1)
        env1 = Envelope()
        env2 = Envelope()
        env1.polarization.uid = "p1"
        env2.polarization.uid = "p2"
        env1.fock.uid = "f1"
        env2.fock.uid = "f2"
        env1.polarization.state = PolarizationLabel.V
        env2.polarization.state = PolarizationLabel.R

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)

        outcomes = ce.measure(env1.polarization, env2.polarization)
        self.assertTrue(isinstance(outcomes, dict))
        for state, outcome in outcomes.items():
            self.assertTrue(isinstance(state, Fock) or isinstance(state, Polarization))

        """
        When measuring a state object, which is a part of an envelope (Fock or Polarization)
        also the other corresponding state object will get measured.
        """

        self.assertEqual(outcomes[env1.polarization], 1)
        self.assertEqual(outcomes[env2.polarization], 0)
        self.assertEqual(outcomes[env1.fock], 0)
        self.assertEqual(outcomes[env2.fock], 0)


        self.assertIsNone(env1.polarization.index)
        self.assertIsNone(env1.polarization.state)
        self.assertIsNone(env2.polarization.index)
        self.assertIsNone(env2.polarization.state)
        self.assertIsNone(env1.composite_envelope_id)
        self.assertIsNone(env2.composite_envelope_id)
        self.assertTrue(env1.measured)
        self.assertTrue(env2.measured)

    def test_measuring_pairs_in_same_product_state(self) -> None:
        """
        When a state object from an enelope is in a product state with
        envelopes other state object, the other state object should also
        be measured
        """
        C = Config()
        C.set_seed(1)
        env1 = Envelope()
        env1.uid = "e1"
        env1.fock.state = 2
        env1.polarization.state = PolarizationLabel.R
        env2 = Envelope()
        env2.uid = "e2"
        env1.polarization.uid = "p1"
        env2.polarization.uid = "p2"
        env1.fock.uid = "f1"
        env2.fock.uid = "f2"
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.polarization, env2.polarization, env1.fock)

        outcomes = ce.measure(env1.polarization)
        self.assertEqual(outcomes[env1.polarization], 1)
        self.assertEqual(outcomes[env1.fock], 2)

        self.assertIsNone(env1.polarization.state)
        self.assertIsNone(env1.fock.index)
        self.assertIsNone(env1.fock.state)
        self.assertTrue(env1.polarization.measured)
        self.assertTrue(env1.fock.measured)
        self.assertTrue(env1.measured)
        self.assertFalse(env2.polarization.measured)
        self.assertFalse(env2.fock.measured)
        self.assertIsNotNone(env2.polarization.index)
        self.assertEqual(env2.polarization.index, (0,0))

    def test_measuring_in_two_product_spaces(self) -> None:
        C = Config()
        C.set_seed(1)
        env1 = Envelope()
        env1.uid = "e1"
        env1.fock.state = 1
        env1.fock.dimensions = 2
        env1.polarization.state = PolarizationLabel.R
        env2 = Envelope()
        env2.uid = "e2"
        env2.fock.dimensions =2
        env1.polarization.uid = "p1"
        env2.polarization.uid = "p2"
        env1.fock.uid = "f1"
        env2.fock.uid = "f2"
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.polarization, env2.polarization)
        ce.combine(env1.fock, env2.fock)
        self.assertEqual(env1.polarization.index, (0,0))
        self.assertEqual(env2.polarization.index, (0,1))
        self.assertEqual(env1.fock.index, (1,0))
        self.assertEqual(env2.fock.index, (1,1))

        outcomes = ce.measure(env1.polarization, env1.fock)
        self.assertEqual(outcomes[env1.polarization], 1)
        self.assertEqual(outcomes[env1.fock], 1)
        self.assertTrue(env1.polarization.measured)
        self.assertTrue(env1.fock.measured)
        self.assertIsNone(env1.polarization.index)
        self.assertIsNone(env1.fock.index)

        outcomes = ce.measure(env2.polarization)
        self.assertEqual(outcomes[env2.polarization], 0)
        self.assertEqual(outcomes[env2.fock], 0)
        self.assertTrue(env2.polarization.measured)
        self.assertTrue(env2.fock.measured)
        self.assertIsNone(env2.polarization.index)
        self.assertIsNone(env2.fock.index)

    def test_measuring_from_envelope(self) -> None:
        C = Config()
        C.set_seed(1)
        env1 = Envelope()
        env1.uid = "e1"
        env1.fock.state = 1
        env1.fock.dimensions = 2
        env1.polarization.state = PolarizationLabel.H
        env2 = Envelope()
        env2.uid = "e2"
        env2.fock.dimensions =2
        env1.polarization.uid = "p1"
        env2.polarization.uid = "p2"
        env1.fock.uid = "f1"
        env2.fock.uid = "f2"
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.polarization, env2.polarization)
        ce.combine(env1.fock, env2.fock)
        outcomes = env1.measure()
        self.assertEqual(outcomes[env1.polarization], 0)
        self.assertEqual(outcomes[env1.fock], 1)
        self.assertTrue(env1.measured)
        self.assertTrue(env1.polarization.measured)
        self.assertTrue(env1.fock.measured)

    def test_measuring_twice(self) -> None:
        C = Config()
        C.set_seed(120)

        env1 = Envelope()
        env1.uid = "e1"
        env1.fock.state = 5
        env1.fock.dimensions = 10
        env1.polarization.state = PolarizationLabel.R
        env2 = Envelope()
        env2.uid = "e2"
        env2.fock.dimensions =8
        env2.fock.state = 7
        env2.polarization.state = PolarizationLabel.L

        env1.polarization.uid = "p1"
        env2.polarization.uid = "p2"
        env1.fock.uid = "f1"
        env2.fock.uid = "f2"
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.polarization, env1.fock, env2.polarization, env2.fock)

        outcomes = ce.measure(env1.polarization)
        self.assertEqual(outcomes[env1.polarization], 1)
        self.assertEqual(outcomes[env1.fock], 5)
        outcomes = ce.measure(env2.polarization)
        self.assertEqual(outcomes[env2.polarization], 0)
        self.assertEqual(outcomes[env2.fock], 7)

    def test_measuring_from_fock(self) -> None:
        C = Config()
        C.set_seed(120)

        env1 = Envelope()
        env1.uid = "e1"
        env1.fock.state = 5
        env1.fock.dimensions = 10
        env1.polarization.state = PolarizationLabel.V
        env2 = Envelope()
        env2.uid = "e2"
        env2.fock.dimensions =8
        env2.fock.state = 7
        env2.polarization.label = PolarizationLabel.L

        env1.polarization.uid = "p1"
        env2.polarization.uid = "p2"
        env1.fock.uid = "f1"
        env2.fock.uid = "f2"
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.polarization, env1.fock, env2.polarization, env2.fock)

        outcomes = env1.fock.measure()
        self.assertEqual(outcomes[env1.polarization], 1)
        self.assertEqual(outcomes[env1.fock], 5)
        outcomes = ce.measure(env2.polarization)
        self.assertEqual(outcomes[env2.polarization], 0)
        self.assertEqual(outcomes[env2.fock], 7)

    def test_measuring_from_polarization(self) -> None:
        C = Config()
        C.set_seed(120)

        env1 = Envelope()
        env1.uid = "e1"
        env1.fock.state = 5
        env1.fock.dimensions = 10
        env1.polarization.state= PolarizationLabel.V
        env2 = Envelope()
        env2.uid = "e2"
        env2.fock.dimensions =8
        env2.fock.state = 7
        env2.polarization.state = PolarizationLabel.L

        env1.polarization.uid = "p1"
        env2.polarization.uid = "p2"
        env1.fock.uid = "f1"
        env2.fock.uid = "f2"
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.polarization, env1.fock, env2.polarization, env2.fock)

        outcomes = env1.polarization.measure()
        self.assertEqual(outcomes[env1.polarization], 1)
        self.assertEqual(outcomes[env1.fock], 5)
        outcomes = ce.measure(env2.polarization)
        self.assertEqual(outcomes[env2.polarization], 0)
        self.assertEqual(outcomes[env2.fock], 7)


class TestCompositeEnvelopeMeasurementsMatrix(unittest.TestCase):
    def atest_simple_measurement(self) -> None:
        env1 = Envelope()
        env2 = Envelope()

        env1.fock.label = 2
        env1.fock.dimensions = 3
        env2.fock.label = 3
        env2.fock.dimensions = 4

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.fock)
        ce.product_states[0].expand()
        outcomes = ce.measure(env1.fock)
        self.assertEqual(outcomes[env1.fock], 2)
        self.assertEqual(outcomes[env1.polarization], 0)
        self.assertTrue(env1.fock.measured)
        self.assertTrue(env1.polarization.measured)
        self.assertTrue(env1.measured)
        outcomes = ce.measure(env2.fock)
        self.assertEqual(outcomes[env2.fock], 3)
        self.assertEqual(outcomes[env2.polarization], 0)
        self.assertTrue(env2.fock.measured)
        self.assertTrue(env2.polarization.measured)
        self.assertTrue(env2.measured)

    def atest_big_product_state_measurement(self) -> None:
        C = Config()
        C.set_seed(120)
        env1 = Envelope()
        env2 = Envelope()
        env3 = Envelope()

        env1.fock.uid = "f1"
        env2.fock.uid = "f2"
        env3.fock.uid = "f3"

        env1.polarization.uid = "p1"
        env2.polarization.uid = "p2"
        env3.polarization.uid = "p3"

        env1.fock.label = 1
        env2.fock.label = 2
        env3.fock.label = 3
        env1.fock.dimensions = 2
        env2.fock.dimensions = 3
        env3.fock.dimensions = 4
        env1.polarization.label  = PolarizationLabel.H
        env2.polarization.label  = PolarizationLabel.R
        env3.polarization.label  = PolarizationLabel.V

        ce = CompositeEnvelope(env1,env2,env3)
        ce.combine(env1.polarization, env2.polarization, env3.polarization, env1.fock, env2.fock)
        ce.product_states[0].expand()

        outcomes = env1.measure()
        self.assertEqual(outcomes[env1.fock], 1)
        self.assertEqual(outcomes[env1.polarization], 0)

        outcomes = env2.measure()
        self.assertEqual(outcomes[env2.fock], 2)
        self.assertEqual(outcomes[env2.polarization], 1)

        outcomes = env3.measure()
        self.assertEqual(outcomes[env3.fock], 3)
        self.assertEqual(outcomes[env3.polarization], 1)
        self.assertEqual(ce.product_states, [])


class TestKrausApply(unittest.TestCase):
    def get_the_last_kraus_operator(self, operators: List[jnp.ndarray]):
        dim = operators[0].shape[0]
        identity = jnp.eye(dim)

        sum_operators = jnp.zeros((dim,dim))
        for op in operators:
            sum_operators += jnp.matmul(op.T.conj(), op)

        remaining = identity - sum_operators
        assert jnp.all(jnp.linalg.eigvals(remaining)>=0)
        last_kraus = jax.scipy.linalg.sqrtm(remaining)
        return last_kraus

    def atest_kraus_apply_vector_full(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env2.fock.dimensions = 2

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)

        op = jnp.array(
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [1,0,0,0]]
        )

        last_op = self.get_the_last_kraus_operator([op])
        operators  = [op, last_op]
        ce.apply_kraus(operators, env1.fock, env2.fock)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0],[0],[0],[1]]
                )
            )
        )

    def atest_kraus_apply_vector_partial(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env3 = Envelope()
        env1.fock.dimensions = 2
        env1.fock.uid = "f1"
        env2.fock.dimensions = 2
        env2.fock.uid = "f2"
        env3.fock.dimensions = 2
        env3.fock.uid = "f3"

        ce = CompositeEnvelope(env1,env2,env3)
        ce.combine(env1.fock, env2.fock, env3.fock)

        op = jnp.array(
            [[0,0],
             [1,0]]
        )
        last_op = self.get_the_last_kraus_operator([op])

        ce.apply_kraus([op,last_op], env3.fock)
        self.assertEqual(
            [env3.fock, env2.fock, env1.fock],
            ce.product_states[0].state_objs
        )
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0],[0],[0],[0],[1],[0],[0],[0]]
                ),
            )
        )

    def atest_kraus_apply_with_two_product_states(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env1.fock.uid = "f1"
        env1.polarization.uid = "p1"
        env2.fock.dimensions = 2
        env2.fock.uid = "f2"
        env2.polarization.uid = "p2"

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.polarization)
        ce.combine(env1.polarization, env2.fock)

        op = jnp.array(
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [1,0,0,0]]
        )
        last_op = self.get_the_last_kraus_operator([op])

        ce.apply_kraus([op, last_op], env1.fock, env2.fock)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[0],[0],[0]])
            )
        )

    def atest_kraus_apply_matrix_full(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env1.fock.uid = "f1"
        env1.fock.label = 1
        env2.fock.dimensions = 2
        env2.fock.uid = "f2"
        env2.fock.label = 1
        env1.fock.expand()
        env1.fock.expand()

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)

        op = jnp.array(
            [[0,0,0,1],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]]
        )
        last_op = self.get_the_last_kraus_operator([op])
        C = Config()
        C.set_contraction(False)

        ce.apply_kraus([op, last_op], env1.fock, env2.fock)
        
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[1,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]]
                )
            )
        )

    def atest_kraus_apply_matrix_full_contraction(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env1.fock.uid = "f1"
        env1.fock.label = 1
        env2.fock.dimensions = 2
        env2.fock.uid = "f2"
        env2.fock.label = 1
        env1.fock.expand()
        env1.fock.expand()

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)

        op = jnp.array(
            [[0,0,0,1],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]]
        )
        last_op = self.get_the_last_kraus_operator([op])
        C = Config()
        C.set_contraction(True)

        ce.apply_kraus([op, last_op], env1.fock, env2.fock)
        
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[1],
                     [0],
                     [0],
                     [0]]
                )
            )
        )

    def atest_kraus_apply_matrix_partial(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env1.fock.uid = "f1"
        env1.fock.label = 1
        env2.fock.dimensions = 2
        env2.fock.uid = "f2"
        env2.fock.label = 1
        env1.fock.expand()
        env1.fock.expand()

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)

        op = jnp.array(
            [[0,1],
             [0,0]]
        )
        last_op = self.get_the_last_kraus_operator([op])
        C = Config()
        C.set_contraction(False)

        ce.apply_kraus([op, last_op], env1.fock)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0,0,0,0],
                     [0,1,0,0],
                     [0,0,0,0],
                     [0,0,0,0]]
                )
            )
        )

    def atest_kraus_apply_with_two_product_states_matrix(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env1.fock.uid = "f1"
        env1.fock.label = 1
        env2.fock.dimensions = 2
        env2.fock.uid = "f2"
        env2.fock.label = 1
        env1.fock.expand()
        env1.fock.expand()

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock)
        ce.combine(env2.fock)

        op = jnp.array(
            [[0,0,0,1],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]]
        )
        last_op = self.get_the_last_kraus_operator([op])
        C = Config()
        C.set_contraction(False)

        ce.apply_kraus([op, last_op], env1.fock, env2.fock)
        
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[1,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]]
                )
            )
        )

    def atest_kraus_exception_dimensions(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env1.fock.uid = "f1"
        env1.fock.label = 1
        env2.fock.dimensions = 2
        env2.fock.uid = "f2"
        env2.fock.label = 1
        env1.fock.expand()
        env1.fock.expand()

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock)
        ce.combine(env2.fock)

        op = jnp.array(
            [[0,0,0,0,1],
             [0,0,0,0,0],
             [0,0,0,0,0],
             [0,0,0,0,0]]
        )
        C = Config()
        C.set_contraction(False)

        with self.assertRaises(ValueError) as context:
            ce.apply_kraus([op], env1.fock, env2.fock)

    def atest_kraus_exception_identity(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env1.fock.uid = "f1"
        env1.fock.label = 1
        env2.fock.dimensions = 2
        env2.fock.uid = "f2"
        env2.fock.label = 1
        env1.fock.expand()
        env1.fock.expand()

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock)
        ce.combine(env2.fock)

        op = jnp.array(
            [[0,0,0,1],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]]
        )
        C = Config()
        C.set_contraction(False)

        with self.assertRaises(ValueError) as context:
            ce.apply_kraus([op], env1.fock, env2.fock)

class TestPOVMMeasurement(unittest.TestCase):
    """
    Testing POVM Measurement Scenarios
    """
    def atest_full_POVM_measurement(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env2.fock.dimensions = 2

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)

        operators = []
        for i in range(4):
            op = jnp.zeros((4,4))
            op = op.at[tuple([i, i])].set(1)  # Convert [i,i] to tuple
            operators.append(op)

        outcomes = ce.measure_POVM(operators, env1.fock, env2.fock)
        self.assertEqual(0, outcomes[0])
        self.assertEqual(
            outcomes[1][env1.polarization], 0
        )
        self.assertEqual(
            outcomes[1][env2.polarization], 0
        )

    def atest_partial_POVM_measurement(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env2.fock.dimensions = 2
        env1.polarization.label = PolarizationLabel.R
        env1.fock.label = 1
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)

        operators = []
        for i in range(2):
            op = jnp.zeros((2,2))
            op = op.at[tuple([i,i])].set(1)
            operators.append(op)

        C = Config()
        C.set_seed = 100
        C.set_contraction(False)
        outcomes = ce.measure_POVM(operators, env1.fock)
        self.assertEqual(1, outcomes[0])
        self.assertEqual(1, outcomes[1][env1.polarization])

    def atest_partial_POVM_measurement_contract(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env2.fock.dimensions = 2
        env1.polarization.label = PolarizationLabel.R
        env1.fock.label = 1
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)

        operators = []
        for i in range(2):
            op = jnp.zeros((2,2))
            op = op.at[tuple([i,i])].set(1)
            operators.append(op)

        C = Config()
        C.set_seed = 100
        C.set_contraction(True)
        outcomes = ce.measure_POVM(operators, env1.fock)
        self.assertEqual(1, outcomes[0])
        self.assertEqual(1, outcomes[1][env1.polarization])
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array([[1],[0]])
            )
        )

    def atest_partial_POVM_measurement_non_destructive(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env2.fock.dimensions = 2
        env1.polarization.label = PolarizationLabel.R
        env1.fock.label = 1
        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)

        operators = []
        for i in range(2):
            op = jnp.zeros((2,2))
            op = op.at[tuple([i,i])].set(1)
            operators.append(op)

        C = Config()
        C.set_seed = 100
        C.set_contraction(False)
        outcomes = ce.measure_POVM(operators, env1.fock, destructive=False)
        self.assertEqual(1, outcomes[0])
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0,0,0,0],
                     [0,0,0,0],
                     [0,0,1,0],
                     [0,0,0,0]]
                )
            )
        )

    def atest_partial_POVM_measurement_superposition(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env2.fock.dimensions = 2
        env1.polarization.label = PolarizationLabel.R
        env1.fock.expand()
        env1.fock.state_vector = env1.fock.state_vector.at[tuple([0,0])].set(0.5)
        env1.fock.state_vector = env1.fock.state_vector.at[tuple([1,0])].set(0.5)

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)

        operators = []
        for i in range(2):
            op = jnp.zeros((2,2))
            op = op.at[tuple([i,i])].set(1)
            operators.append(op)

        C = Config()
        C.set_seed = 100
        C.set_contraction(False)
        outcomes = ce.measure_POVM(operators, env1.fock, destructive=False)
        self.assertEqual(1, outcomes[0])
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0,0,0,0],
                     [0,0,0,0],
                     [0,0,1,0],
                     [0,0,0,0]]
                )
            )
        )

    def atest_POVM_two_product_spaces(self) -> None:
        env1 = Envelope()
        env1.polarization.uid = "p1"
        env1.fock.uid = "f1"
        env2 = Envelope()
        env2.polarization.uid = "p2"
        env2.fock.uid = "f2"
        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.fock)
        ce.combine(env1.fock, env2.polarization)

        operators = []
        for i in range(4):
            op = jnp.zeros((4,4))
            op = op.at[tuple([i,i])].set(1)
            operators.append(op)

        outcome = ce.measure_POVM(operators, env1.polarization, env2.polarization)
        self.assertEqual(outcome[0], 0)

    def atest_POVM_exception(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env2.fock.dimensions = 2
        env1.polarization.label = PolarizationLabel.R

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.fock, env2.fock)

        operators = []
        for i in range(2):
            op = jnp.zeros((3,3))
            op = op.at[tuple([i,i])].set(1)
            operators.append(op)

        C = Config()
        C.set_seed = 100
        C.set_contraction(False)
        with self.assertRaises(ValueError) as context:
            outcomes = ce.measure_POVM(operators, env1.fock, destructive=False)



class TestCompositeMatrixTrace(unittest.TestCase):
    def atest_trace_vector(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env3 = Envelope()
        env1.polarization.label = PolarizationLabel.R
        env2.polarization.label = PolarizationLabel.V
        env3.polarization.label = PolarizationLabel.H


        ce = CompositeEnvelope(env1,env2,env3)
        ce.combine(env1.polarization, env2.polarization, env3.polarization)

        to = ce.trace_out(env1.polarization, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                to,
                jnp.array(
                    [[0],
                     [1/jnp.sqrt(2)],
                     [0],
                     [1j/jnp.sqrt(2)]]
                )
            )
        )

        # Check that the state is not changed
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
                     [1j/jnp.sqrt(2)],
                     [0]]
                )
            )
        )

    def atest_trace_vector_two_product_states(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env3 = Envelope()
        env1.polarization.label = PolarizationLabel.R
        env2.polarization.label = PolarizationLabel.V
        env3.polarization.label = PolarizationLabel.H


        ce = CompositeEnvelope(env1,env2,env3)
        ce.combine(env1.polarization, env3.polarization)
        ce.combine(env2.polarization, env2.fock)

        to = ce.trace_out(env1.polarization, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                to,
                jnp.array(
                    [[0],
                     [1/jnp.sqrt(2)],
                     [0],
                     [1j/jnp.sqrt(2)]]
                )
            )
        )

    def atest_trace_matrix(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env3 = Envelope()
        env1.polarization.label = PolarizationLabel.R
        env2.polarization.label = PolarizationLabel.V
        env3.polarization.label = PolarizationLabel.H


        ce = CompositeEnvelope(env1,env2,env3)
        env1.polarization.expand()
        env1.polarization.expand()
        ce.combine(env1.polarization, env2.polarization, env3.polarization)

        to = ce.trace_out(env1.polarization, env2.polarization)
        self.assertTrue(
            jnp.allclose(
                to,
                jnp.array(
                    [[0,0,0,0],
                     [0,0.5,0,-0.5j],
                     [0,0,0,0],
                     [0,0.5j,0,0.5]]
                )
            )
        )
        # Check that the state is not changed
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0.5,0,0,0,-0.5j,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0],
                     [0,0,0.5j,0,0,0,0.5,0],
                     [0,0,0,0,0,0,0,0]]
                )
            )
        )


class TestContraction(unittest.TestCase):
    def atest_contraction(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.polarization.label = PolarizationLabel.H
        env2.polarization.label = PolarizationLabel.V

        ce = CompositeEnvelope(env1,env2)
        ce.combine(env1.polarization, env2.polarization)
        ce.product_states[0].expand()
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0,0,0,0],
                     [0,1,0,0],
                     [0,0,0,0],
                     [0,0,0,0]])
            )
        )
        ce.product_states[0].contract()
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0],
                     [1],
                     [0],
                     [0]])
            )
        )

    def atest_contraction_fail(self) -> None:
        """
        Case when contraction should not be possible
        """
        env1 = Envelope()
        env2 = Envelope()

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.polarization, env2.polarization)
        container = CompositeEnvelope._containers[ce.uid]
        rho = jnp.array(
            [[0.5, 0.1, 0.0, 0.0],
             [0.1, 0.3, 0.0, 0.0],
             [0.0, 0.0, 0.2, 0.1],
             [0.0, 0.0, 0.1, 0.1]])

        container.states[0].expansion_level = ExpansionLevel.Matrix
        container.states[0].state = rho.copy()
        container.states[0].contract()
        self.assertTrue(
            jnp.allclose(
                container.states[0].state,
                rho
            )
        )
        self.assertEqual(container.states[0].expansion_level, ExpansionLevel.Matrix)
