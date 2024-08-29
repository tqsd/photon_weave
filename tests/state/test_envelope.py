import unittest
from typing import Union
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
        env.fock.label = 1
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
        env.measure()
        self.assertEqual(env.__repr__(), "Envelope already measured")

        env = Envelope()
        env.fock.label = 1
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
        env.fock.label = 1
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
        for item in [env.composite_envelope, env.composite_matrix, env.composite_vector]:
            self.assertIsNone(item)
        self.assertEqual(env.wavelength, 1550)
        self.assertTrue(isinstance(env.temporal_profile, TemporalProfileInstance))
        self.assertEqual(env.temporal_profile.params["mu"], 0)


        fock = Fock()
        env = Envelope(fock=fock)
        self.assertTrue(isinstance(env.fock, Fock))
        self.assertTrue(fock is env.fock)
        self.assertTrue(isinstance(env.polarization, Polarization))
        for item in [env.composite_envelope, env.composite_matrix, env.composite_vector]:
            self.assertIsNone(item)
        self.assertEqual(env.wavelength, 1550)
        self.assertTrue(isinstance(env.temporal_profile, TemporalProfileInstance))
        self.assertEqual(env.temporal_profile.params["mu"], 0)

        pol = Polarization()
        env = Envelope(polarization=pol)
        self.assertTrue(isinstance(env.fock, Fock))
        self.assertTrue(isinstance(env.polarization, Polarization))
        for item in [env.composite_envelope, env.composite_matrix, env.composite_vector]:
            self.assertIsNone(item)
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
        for item in [env.composite_envelope, env.composite_matrix, env.composite_vector]:
            self.assertIsNone(item)
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
        get_vals = lambda x: (x.label, x.state_vector, x.density_matrix)
        for item in get_vals(pol):
            self.assertIsNone(item)
        for item in get_vals(fock):
            self.assertIsNone(item)
        self.assertEqual(fock.index, 0)
        self.assertEqual(pol.index, 1)
        self.assertIsNone(env.composite_matrix)
        self.assertTrue(
            jnp.allclose(
                env.composite_vector,
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
        get_vals = lambda x: (x.label, x.state_vector, x.density_matrix)
        for item in get_vals(pol):
            self.assertIsNone(item)
        for item in get_vals(fock):
            self.assertIsNone(item)
        self.assertEqual(fock.index, 0)
        self.assertEqual(pol.index, 1)
        self.assertIsNone(env.composite_matrix)
        self.assertTrue(
            jnp.allclose(
                env.composite_vector,
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
        get_vals = lambda x: (x.label, x.state_vector, x.density_matrix)
        for item in get_vals(pol):
            self.assertIsNone(item)
        for item in get_vals(fock):
            self.assertIsNone(item)
        self.assertEqual(fock.index, 0)
        self.assertEqual(pol.index, 1)
        self.assertIsNone(env.composite_matrix)
        self.assertTrue(
            jnp.allclose(
                env.composite_vector,
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
        get_vals = lambda x: (x.label, x.state_vector, x.density_matrix)
        for item in get_vals(pol):
            self.assertIsNone(item)
        for item in get_vals(fock):
            self.assertIsNone(item)
        self.assertEqual(fock.index, 0)
        self.assertEqual(pol.index, 1)
        self.assertIsNone(env.composite_vector)
        self.assertTrue(
            jnp.allclose(
                env.composite_matrix,
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
        get_vals = lambda x: (x.label, x.state_vector, x.density_matrix)
        for item in get_vals(pol):
            self.assertIsNone(item)
        for item in get_vals(fock):
            self.assertIsNone(item)
        self.assertEqual(fock.index, 0)
        self.assertEqual(pol.index, 1)
        self.assertIsNone(env.composite_vector)
        self.assertTrue(
            jnp.allclose(
                env.composite_matrix,
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
        self.assertEqual(env.expansion_level, -1)
        env.combine()
        self.assertEqual(env.expansion_level, ExpansionLevel.Vector)
        env.expand()
        self.assertEqual(env.expansion_level, ExpansionLevel.Matrix)


        env = Envelope()
        env.expand()
        self.assertIsNone(env.polarization.label)
        self.assertIsNone(env.fock.label)
        self.assertIsNone(env.polarization.density_matrix)
        self.assertIsNone(env.fock.density_matrix)
        env.expand()
        self.assertIsNone(env.polarization.label)
        self.assertIsNone(env.fock.label)
        self.assertIsNone(env.polarization.state_vector)
        self.assertIsNone(env.fock.state_vector)


class TestEnvelopeMeausrement(unittest.TestCase):
    def test_measurement_separate(self) -> None:
        env = Envelope()
        m = env.measure()
        self.assertEqual(m, (0,0))

        pol = Polarization(PolarizationLabel.V)
        fock = Fock()
        fock.label = 1
        env = Envelope(polarization=pol, fock=fock)
        m = env.measure()
        self.assertEqual(m, (1,1))
        

    def test_measurement_combined_vector(self) -> None:
        env = Envelope()
        env.combine()
        m = env.measure()
        self.assertEqual(m, (0,0))

        pol = Polarization(PolarizationLabel.V)
        fock = Fock()
        fock.label = 1
        env = Envelope(polarization=pol, fock=fock)
        env.combine()
        m = env.measure()
        self.assertEqual(m, (1,1))

        C = Config()
        C.set_seed(1)

        pol = Polarization(PolarizationLabel.R)
        fock = Fock()
        fock.label = 1
        env = Envelope(polarization=pol, fock=fock)
        env.combine()
        m = env.measure()
        self.assertEqual(m, (1,0), "Should measure (1,0) wiht the seed 1")

    def test_measurement_combined_matrix(self) -> None:
        C = Config()
        env=Envelope()
        env.combine()
        env.expand()
        m = env.measure()
        self.assertEqual(m, (0,0))

        C.set_seed(1)
        env=Envelope()
        env.fock.label = 5
        env.fock.dimensions = 10
        env.polarization.label = PolarizationLabel.R
        env.combine()
        env.expand()
        m = env.measure()
        self.assertTrue(env.measured)
        self.assertEqual(m, (5,0))
        with self.assertRaises(EnvelopeAlreadyMeasuredException):
            env.measure()
