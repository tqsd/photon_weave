import unittest
from typing import Union
import numpy as np
import jax.numpy as jnp

from photon_weave.photon_weave import Config
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.envelope import Envelope, TemporalProfile, TemporalProfileInstance
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization, PolarizationLabel
from photon_weave.state.exceptions import (
    EnvelopeAlreadyMeasuredException,
    EnvelopeAssignedException,
    MissingTemporalProfileArgumentException
)

class TestEnvelopeSmallFunctions(unittest.TestCase):
    def test_envelope_repr(self):
        """
        Test the representation of envelope
        """
        env = Envelope()
        self.assertEqual("|0⟩ ⊗ |H⟩",env.__repr__())
        env.polarization.expand()

        representation = env.__repr__().split("\n")
        representation = [r.split(" ⊗ ") for r in representation]
        self.assertEqual(representation[0][0], "|0⟩")
        self.assertEqual(representation[0][1], "⎡ 1.00 + 0.00j ⎤")
        self.assertEqual(representation[1][1], "⎣ 0.00 + 0.00j ⎦")

        env.polarization.expand()
        representation = env.__repr__().split("\n")
        representation = [r.split(" ⊗ ") for r in representation]
        self.assertEqual(representation[0][0], "|0⟩")
        self.assertEqual(representation[0][1], "⎡ 1.00 + 0.00j   0.00 + 0.00j ⎤")
        self.assertEqual(representation[1][1], "⎣ 0.00 + 0.00j   0.00 + 0.00j ⎦")

        env.fock.expand()
        representation = env.__repr__().split("\n")
        representation = [r.split(" ⊗ ") for r in representation]
        self.assertEqual(representation[0][0],"⎡ 1.00 + 0.00j ⎤" )
        self.assertEqual(representation[1][0],"⎢ 0.00 + 0.00j ⎥" )
        self.assertEqual(representation[2][0],"⎣ 0.00 + 0.00j ⎦" )
        self.assertEqual(representation[0][1],"⎡ 1.00 + 0.00j   0.00 + 0.00j ⎤" )
        self.assertEqual(representation[1][1],"⎣ 0.00 + 0.00j   0.00 + 0.00j ⎦" )

        env.fock.expand()
        representation = env.__repr__().split("\n")
        representation = [r.split(" ⊗ ") for r in representation]
        self.assertEqual(representation[0][0],"⎡ 1.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎤" )
        self.assertEqual(representation[1][0],"⎢ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎥" )
        self.assertEqual(representation[2][0],"⎣ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎦" )
        self.assertEqual(representation[0][1],"⎡ 1.00 + 0.00j   0.00 + 0.00j ⎤" )
        self.assertEqual(representation[1][1],"⎣ 0.00 + 0.00j   0.00 + 0.00j ⎦" )

        env = Envelope()
        env.fock.state= 1
        env.fock.expand()
        representation = env.__repr__().split("\n")
        representation = [r.split(" ⊗ ") for r in representation]
        self.assertEqual(representation[0][1], "|H⟩")
        self.assertEqual(representation[0][0], "⎡ 0.00 + 0.00j ⎤")
        self.assertEqual(representation[1][0], "⎢ 1.00 + 0.00j ⎥")
        self.assertEqual(representation[2][0], "⎢ 0.00 + 0.00j ⎥")
        self.assertEqual(representation[3][0], "⎣ 0.00 + 0.00j ⎦")
        
        env.fock.expand()
        representation = env.__repr__().split("\n")
        representation = [r.split(" ⊗ ") for r in representation]
        self.assertEqual(representation[0][1], "|H⟩")
        self.assertEqual(representation[0][0], "⎡ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎤")
        self.assertEqual(representation[1][0], "⎢ 0.00 + 0.00j   1.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎥")
        self.assertEqual(representation[2][0], "⎢ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎥")
        self.assertEqual(representation[3][0], "⎣ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎦")

        env.polarization.expand()
        representation = env.__repr__().split("\n")
        representation = [r.split(" ⊗ ") for r in representation]

        self.assertEqual(representation[0][0], "⎡ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎤")
        self.assertEqual(representation[1][0], "⎢ 0.00 + 0.00j   1.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎥")
        self.assertEqual(representation[2][0], "⎢ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎥")
        self.assertEqual(representation[3][0], "⎣ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎦")
        self.assertEqual(representation[0][1], "⎡ 1.00 + 0.00j ⎤")
        self.assertEqual(representation[1][1], "⎣ 0.00 + 0.00j ⎦")

        env.polarization.expand()
        representation = env.__repr__().split("\n")
        representation = [r.split(" ⊗ ") for r in representation]
        self.assertEqual(representation[0][0], "⎡ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎤")
        self.assertEqual(representation[1][0], "⎢ 0.00 + 0.00j   1.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎥")
        self.assertEqual(representation[2][0], "⎢ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎥")
        self.assertEqual(representation[3][0], "⎣ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎦")
        self.assertEqual(representation[0][1], "⎡ 1.00 + 0.00j   0.00 + 0.00j ⎤" )
        self.assertEqual(representation[1][1], "⎣ 0.00 + 0.00j   0.00 + 0.00j ⎦" )

        env = Envelope()
        env.fock.state = 1
        env.combine()
        representation = env.__repr__().split("\n")
        self.assertEqual(representation[0], "⎡ 0.00 + 0.00j ⎤")
        self.assertEqual(representation[1], "⎢ 0.00 + 0.00j ⎥")
        self.assertEqual(representation[2], "⎢ 1.00 + 0.00j ⎥")
        self.assertEqual(representation[3], "⎢ 0.00 + 0.00j ⎥")
        self.assertEqual(representation[4], "⎢ 0.00 + 0.00j ⎥")
        self.assertEqual(representation[5], "⎢ 0.00 + 0.00j ⎥")
        self.assertEqual(representation[6], "⎢ 0.00 + 0.00j ⎥")
        self.assertEqual(representation[7], "⎣ 0.00 + 0.00j ⎦")

        env = Envelope()
        env.fock.state = 1
        env.fock.dimensions = 2
        env.fock.expand()
        env.fock.expand()
        env.combine()
        representation = env.__repr__().split("\n")
        self.assertEqual(representation[0], "⎡ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎤")
        self.assertEqual(representation[1], "⎢ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎥")
        self.assertEqual(representation[2], "⎢ 0.00 + 0.00j   0.00 + 0.00j   1.00 + 0.00j   0.00 + 0.00j ⎥")
        self.assertEqual(representation[3], "⎣ 0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j   0.00 + 0.00j ⎦")

    def test_initialization(self):
        """
        Test envelope initiaizations
        """
        env = Envelope()
        self.assertTrue(isinstance(env.fock, Fock))
        self.assertTrue(isinstance(env.polarization, Polarization))
        self.assertIsNone(env.state)
        self.assertEqual(env.wavelength, 1550)
        self.assertTrue(isinstance(env.temporal_profile, TemporalProfileInstance))
        self.assertEqual(env.temporal_profile.params["mu"], 0)


        fock = Fock()
        env = Envelope(fock=fock)
        self.assertTrue(isinstance(env.fock, Fock))
        self.assertTrue(fock is env.fock)
        self.assertTrue(isinstance(env.polarization, Polarization))
        self.assertIsNone(env.state)
        self.assertEqual(env.wavelength, 1550)
        self.assertTrue(isinstance(env.temporal_profile, TemporalProfileInstance))
        self.assertEqual(env.temporal_profile.params["mu"], 0)

        pol = Polarization()
        env = Envelope(polarization=pol)
        self.assertTrue(isinstance(env.fock, Fock))
        self.assertTrue(isinstance(env.polarization, Polarization))
        self.assertIsNone(env.state)
        self.assertEqual(env.wavelength, 1550)
        self.assertTrue(isinstance(env.temporal_profile, TemporalProfileInstance))
        self.assertEqual(env.temporal_profile.params["mu"], 0)

        pol = Polarization()
        fock = Fock()
        env = Envelope(polarization=pol, fock=fock)
        self.assertTrue(isinstance(env.polarization, Polarization))
        self.assertTrue(isinstance(env.fock, Fock))
        self.assertTrue(fock is env.fock)
        self.assertTrue(isinstance(env.polarization, Polarization))
        self.assertIsNone(env.state)
        self.assertEqual(env.wavelength, 1550)
        self.assertTrue(isinstance(env.temporal_profile, TemporalProfileInstance))
        self.assertEqual(env.temporal_profile.params["mu"], 0)

    def test_state_combining(self) -> None:
        """
        Test if the states are correctly combined
        """
        pol = Polarization()
        fock = Fock()
        env = Envelope(polarization=pol, fock=fock)
        env.combine()
        self.assertIsNone(pol.state)
        self.assertIsNone(fock.state)
        self.assertEqual(fock.index, 0)
        self.assertEqual(pol.index, 1)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[1],[0],[0],[0],[0],[0]]
                )
            )
        )

        pol = Polarization()
        fock = Fock()
        pol.expand()
        env = Envelope(polarization=pol, fock=fock)
        env.combine()
        self.assertIsNone(pol.state)
        self.assertIsNone(fock.state)
        self.assertEqual(fock.index, 0)
        self.assertEqual(pol.index, 1)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[1],[0],[0],[0],[0],[0]]
                )
            )
        )

        pol = Polarization()
        fock = Fock()
        fock.expand()
        env = Envelope(polarization=pol, fock=fock)
        env.combine()
        self.assertIsNone(pol.state)
        self.assertIsNone(fock.state)
        self.assertEqual(fock.index, 0)
        self.assertEqual(pol.index, 1)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[1],[0],[0],[0],[0],[0]]
                )
            )
        )

        pol = Polarization()
        fock = Fock()
        pol.expand()
        pol.expand()
        env = Envelope(polarization=pol, fock=fock)
        env.combine()
        self.assertIsNone(pol.state)
        self.assertIsNone(fock.state)
        self.assertEqual(fock.index, 0)
        self.assertEqual(pol.index, 1)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[1,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]]
                )
            )
        )

        pol = Polarization()
        fock = Fock()
        fock.expand()
        fock.expand()
        env = Envelope(polarization=pol, fock=fock)
        env.combine()
        self.assertIsNone(pol.state)
        self.assertIsNone(fock.state)
        self.assertEqual(fock.index, 0)
        self.assertEqual(pol.index, 1)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[1,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]]
                )
            )
        )

    def test_expansion_levels(self) -> None:
        env = Envelope()
        self.assertEqual(env.expansion_level, ExpansionLevel.Label)
        env.combine()
        self.assertEqual(env.expansion_level, ExpansionLevel.Vector)
        env.expand()
        self.assertEqual(env.expansion_level, ExpansionLevel.Matrix)


        env = Envelope()
        env.expand()
        self.assertEqual(env.expansion_level, ExpansionLevel.Vector)
        self.assertEqual(env.fock.expansion_level, ExpansionLevel.Vector)
        self.assertEqual(env.polarization.expansion_level, ExpansionLevel.Vector)
        env.expand()
        self.assertEqual(env.expansion_level, ExpansionLevel.Matrix)
        self.assertEqual(env.fock.expansion_level, ExpansionLevel.Matrix)
        self.assertEqual(env.polarization.expansion_level, ExpansionLevel.Matrix)

    def test_reorder(self) -> None:
        """
        Test the reorder method, the method should
        reorder the spaces in the tensor (kron) product
        accoring to the parameters
        """
        env=Envelope()
        env.reorder(env.fock, env.polarization)
        # Nothing should change, because the envelope is not combined
        self.assertIsNone(env.fock.index)
        self.assertIsNone(env.polarization.index)
        with self.assertRaises(ValueError) as context:
            env.reorder(Fock())
        with self.assertRaises(ValueError) as context:
            env.reorder(env.fock, env.fock)
        with self.assertRaises(ValueError) as context:
            env.reorder(env.fock, env.polarization, Fock())
        env.fock.state = 1
        env.fock.dimensions = 3
        env.combine()
        test_vector = jnp.copy(env.state)
        env.reorder(env.fock, env.polarization)
        self.assertTrue(
            jnp.allclose(
                env.state,
                test_vector
            )
        )
        env.reorder(env.polarization, env.fock)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[0],[1],[0],[0],[0],[0]]
                )
            )
        )
        self.assertEqual(env.fock.index, 1)
        self.assertEqual(env.polarization.index, 0)

        env.expand()
        env.reorder(env.fock, env.polarization)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,1,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]]
                )
            )
        )
        self.assertEqual(env.fock.index, 0)
        self.assertEqual(env.polarization.index, 1)

        # Reorder with only one parameter given
        env = Envelope()
        env.fock.dimensions = 3
        env.fock.state = 1
        env.combine()
        # should not reorder, since by default fock is first
        env.reorder(env.fock)

        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[0],[0],[1],[0],[0],[0]]
                )
            )
        )
        self.assertEqual(env.fock.index, 0)
        self.assertEqual(env.polarization.index, 1)

        env.reorder(env.polarization)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[0],[1],[0],[0],[0],[0]]
                )
            )
        )
        self.assertEqual(env.fock.index, 1)
        self.assertEqual(env.polarization.index, 0)

        # Try with matrix form
        env.expand()
        env.reorder(env.fock)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,1,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]]
                )
            )
        )
        self.assertEqual(env.fock.index, 0)
        self.assertEqual(env.polarization.index, 1)


class TestEnvelopeMeausrement(unittest.TestCase):
    def test_measurement(self) -> None:
        env = Envelope()
        m = env.measure()
        self.assertEqual(m[env.fock], 0)
        self.assertEqual(m[env.polarization], 0)

        pol = Polarization(PolarizationLabel.V)
        fock = Fock()
        fock.state = 1
        env = Envelope(polarization=pol, fock=fock)
        m = env.measure()
        self.assertEqual(m[env.fock], 1)
        self.assertEqual(m[env.polarization], 1)
        self.assertTrue(env.measured)
        self.assertTrue(env.fock.measured)
        self.assertTrue(env.polarization.measured)
        self.assertIsNone(env.state)
        self.assertIsNone(env.fock.state)
        self.assertIsNone(env.polarization.state)

    def test_measurement_combined_vector(self) -> None:
        env = Envelope()
        env.combine()
        m = env.measure()
        self.assertEqual(m[env.fock], 0)
        self.assertEqual(m[env.polarization], 0)

        pol = Polarization(PolarizationLabel.V)
        fock = Fock()
        fock.state = 1
        env = Envelope(polarization=pol, fock=fock)
        env.combine()
        m = env.measure()
        self.assertEqual(m[env.fock], 1)
        self.assertEqual(m[env.polarization], 1)

        C = Config()
        C.set_seed(1)

        pol = Polarization(PolarizationLabel.R)
        fock = Fock()
        fock.state= 1
        env = Envelope(polarization=pol, fock=fock)
        env.combine()
        m = env.measure()
        self.assertEqual(m[env.fock], 1)
        self.assertEqual(m[env.polarization], 0)

    def test_measurement_combined_vector_reordered(self) -> None:
        env = Envelope()
        env.combine()
        env.reorder(env.polarization, env.fock)
        m = env.measure()
        self.assertEqual(m[env.fock], 0)
        self.assertEqual(m[env.polarization], 0)

        pol = Polarization(PolarizationLabel.V)
        fock = Fock()
        fock.state = 1
        env = Envelope(polarization=pol, fock=fock)
        env.combine()
        m = env.measure()
        self.assertEqual(m[env.fock], 1)
        self.assertEqual(m[env.polarization], 1)

        C = Config()
        C.set_seed(1)

        pol = Polarization(PolarizationLabel.R)
        fock = Fock()
        fock.state= 1
        env = Envelope(polarization=pol, fock=fock)
        env.combine()
        env.reorder(env.polarization, env.fock)
        m = env.measure()
        self.assertEqual(m[env.fock], 1)
        self.assertEqual(m[env.polarization], 0)

    def test_measurement_combined_matrix(self) -> None:
        C = Config()
        env = Envelope()
        env.combine()
        env.expand()
        m = env.measure()
        self.assertEqual(m[env.fock], 0)
        self.assertEqual(m[env.polarization], 0)

        C.set_seed(1)
        env = Envelope()
        env.fock.state = 2
        env.fock.dimensions = 3
        env.polarization.state = PolarizationLabel.R
        env.combine()
        env.expand()
        m = env.measure()
        self.assertTrue(env.measured)
        self.assertEqual(m[env.fock], 2)
        self.assertEqual(m[env.polarization], 0)
        with self.assertRaises(ValueError) as context:
            env.measure()

    def test_measurement_combined_matrix_reordered(self) -> None:
        C = Config()
        env = Envelope()
        env.combine()
        env.expand()
        env.reorder(env.polarization, env.fock)
        m = env.measure()
        self.assertEqual(m[env.fock], 0)
        self.assertEqual(m[env.polarization], 0)

        C.set_seed(1)
        env = Envelope()
        env.fock.state = 2
        env.fock.dimensions = 3
        env.polarization.state = PolarizationLabel.R
        env.combine()
        env.expand()
        env.reorder(env.polarization, env.fock)
        m = env.measure()
        self.assertTrue(env.measured)
        self.assertEqual(m[env.fock], 2)
        self.assertEqual(m[env.polarization], 0)
        with self.assertRaises(ValueError) as context:
            env.measure()

    def test_measurement_destructive_partial_vector(self) -> None:
        C = Config()
        C.set_seed = 1
        env = Envelope()
        env.fock.state = 1
        env.fock.dimensions = 3
        env.polarization.state = PolarizationLabel.V
        env.combine()
        out = env.measure(env.fock, separate_measurement=True)
        self.assertTrue(env.measured, True)
        self.assertEqual(out[env.fock], 1)
        self.assertEqual(env.polarization.state, PolarizationLabel.V)
        env = Envelope()
        env.fock.state = 1
        env.fock.dimensions = 3
        env.polarization.state = PolarizationLabel.V
        env.combine()
        env.reorder(env.polarization, env.fock)
        out = env.measure(env.fock, separate_measurement=True)
        self.assertTrue(env.measured, True)
        self.assertEqual(out[env.fock], 1)
        self.assertEqual(env.polarization.state, PolarizationLabel.V)
        env = Envelope()
        env.fock.state = 1
        env.fock.dimensions = 3
        env.polarization.state = PolarizationLabel.V
        env.combine()
        env.reorder(env.polarization, env.fock)
        out = env.measure(env.polarization, separate_measurement=True)
        self.assertTrue(env.measured, True)
        self.assertEqual(out[env.polarization], 1)
        self.assertEqual(env.fock.state, 1)

    def test_measurement_destructive_partial_matrix(self) -> None:
        C = Config()
        C.set_seed = 1
        env = Envelope()
        env.fock.state = 1
        env.fock.dimensions = 3
        env.polarization.state = PolarizationLabel.V
        env.combine()
        env.expand()
        out = env.measure(env.fock, separate_measurement=True)
        self.assertTrue(env.measured, True)
        self.assertEqual(out[env.fock], 1)
        self.assertEqual(env.polarization.state, PolarizationLabel.V)
        env = Envelope()
        env.fock.state = 1
        env.fock.dimensions = 3
        env.polarization.state = PolarizationLabel.V
        env.combine()
        env.expand()
        env.reorder(env.polarization, env.fock)
        out = env.measure(env.fock, separate_measurement=True)
        self.assertTrue(env.measured, True)
        self.assertEqual(out[env.fock], 1)
        self.assertEqual(env.polarization.state, PolarizationLabel.V)
        env = Envelope()
        env.fock.state = 1
        env.fock.dimensions = 3
        env.polarization.state = PolarizationLabel.V
        env.combine()
        env.expand()
        out = env.measure(env.polarization, separate_measurement=True)
        self.assertTrue(env.measured, True)
        self.assertEqual(out[env.polarization], 1)
        self.assertEqual(env.fock.state, 1)
        env = Envelope()
        env.fock.state = 1
        env.fock.dimensions = 3
        env.polarization.state = PolarizationLabel.V
        env.combine()
        env.expand()
        env.reorder(env.polarization, env.fock)
        out = env.measure(env.polarization, separate_measurement=True)
        self.assertTrue(env.measured, True)
        self.assertEqual(out[env.polarization], 1)
        self.assertEqual(env.fock.state, 1)

    def atest_povm_measurement_exceptions(self) -> None:
        env = Envelope()
        with self.assertRaises(ValueError):
            env.measure_POVM(operators=[], states=[env.fock, env.fock])
        with self.assertRaises(ValueError):
            env.measure_POVM(operators=[], states=[env.polarization, env.polarization])
        with self.assertRaises(ValueError):
            env.measure_POVM(operators=[], states=[env.polarization, env.fock, Fock()])
        with self.assertRaises(ValueError):
            env.measure_POVM(operators = [], states=[Polarization(), Fock()])
        with self.assertRaises(ValueError):
            env.measure_POVM(operators = [], states=[env.fock, Polarization()])
        with self.assertRaises(ValueError):
            env.measure_POVM(operators = [], states=[Fock(), env.polarization])
            
    def atest_POVM_measurement_not_combined(self) -> None:
        """
        Test POVM measurement, when fock and polarization spaces
        are not in a product space.
        """
        env = Envelope()
        env.fock.dimensions = 2
        op1 = jnp.array(
            [[1,0],
             [0,0]]
        )
        op2 = jnp.array(
            [[0,0],
             [0,1]]
        )
        m = env.measure_POVM(operators=[op1, op2], states = [env.polarization])
        self.assertEqual(m, 0)
        m = env.measure_POVM(operators=[op1, op2], states = [env.fock])
        self.assertEqual(m, 0)

    def atest_POVM_measurement_combined_partial(self) -> None:
        env = Envelope()
        env.fock.dimensions = 2
        env.combine()

        op1 = jnp.array(
            [[1,0],
             [0,0]]
        )
        op2 = jnp.array(
            [[0,0],
             [0,1]]
        )
        m = env.measure_POVM(operators=[op1, op2],states= [env.polarization])

    def atest_POVM_measurement_combined_full(self) -> None:
        env = Envelope()
        env.fock.dimensions = 2
        op1 = jnp.array( #|0>|H> -> We should measure this
            [[1,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]]
        )
        op2 = jnp.array( #|1>|H>
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,1,0],
             [0,0,0,0]]
        )
        op3 = jnp.array( #|0>|V>
            [[0,0,0,0],
             [0,1,0,0],
             [0,0,0,0],
             [0,0,0,0]]
        )
        op4 = jnp.array( #|1>|V>
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,1]]
        )
        m = env.measure_POVM(
            operators=[op1,op2,op3,op4],
            states=[env.fock, env.polarization],
            destructive=True)
        self.assertEqual(m,0)

class TestEnvelopeKraus(unittest.TestCase):
    def atest_envelope_kraus_exceptions(self) -> None:
        env = Envelope()

        # Test exception with non unique states given
        with self.assertRaises(ValueError):
            env.apply_kraus(operators=[], states=[env.fock, env.fock])
        with self.assertRaises(ValueError):
            env.apply_kraus(operators=[], states=[env.polarization, env.polarization])

        # Test exception with too many states given
        with self.assertRaises(ValueError):
            env.apply_kraus(operators=[], states=[env.polarization, env.fock, Fock()])

        # Test exception with states that are not part of the envelope
        with self.assertRaises(ValueError):
            env.apply_kraus(operators=[], states=[Polarization(), env.fock])
        with self.assertRaises(ValueError):
            env.apply_kraus(operators=[], states=[env.polarization, Fock()])
        with self.assertRaises(ValueError):
            env.apply_kraus(operators=[], states=[Polarization(), Fock()])

    def atest_envelope_kraus_not_combined(self) -> None:
        """
        Test the application of Quantum Channel (kraus) in
        the case where the envelope is not combined,
        polarization and fock spaces are separate
        """
        env = Envelope()
        env.fock.dimensions=3
        env.apply_kraus(operators=[np.array([[0,1],[1,0]])], states=[env.polarization])
        self.assertEqual(env.polarization.label, PolarizationLabel.V)
        self.assertEqual(env.fock.label, 0)
        op1 = jnp.array([[0,0,0],[1,0,0],[0,0,0]])
        op2 = jnp.array([[0,0,0],[0,0,0],[0,1,0]])
        op3 = jnp.array([[0,0,0],[0,0,0],[0,0,1]])
        env.apply_kraus(operators=[op1, op2, op3], states=[env.fock])
        self.assertEqual(env.polarization.label, PolarizationLabel.V)
        self.assertEqual(env.fock.label, 1)

    def atest_envelope_kraus(self) -> None:
        env = Envelope()
        env.fock.dimensions=2
        op1 = jnp.array([[0,0,0,0],
                         [1,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0]])
        op2 = jnp.array([[0,0,0,0],
                         [0,1,0,0],
                         [0,0,1,0],
                         [0,0,0,1]])
        env.apply_kraus(operators = [op1, op2], states=[env.fock, env.polarization])
        self.assertTrue(
            jnp.allclose(
                env.composite_matrix,
                jnp.array(
                    [[0,0,0,0],
                     [0,1,0,0],
                     [0,0,0,0],
                     [0,0,0,0]]
                )
            )
        )
        self.assertEqual(env.fock.index, 0)
        self.assertEqual(env.polarization.index, 1)

        # Kraus operator dimensions missmatch
        with self.assertRaises(ValueError):
            env.apply_kraus(
                operators = [jnp.array([[0,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])],
                states=[env.fock, env.polarization])

        # Kraus operators do not sum to one
        with self.assertRaises(ValueError):
            env.apply_kraus(
                operators = [op1],
                states=[env.fock, env.polarization])
