import unittest

import jax.numpy as jnp

from photon_weave.photon_weave import Config
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.polarization import Polarization, PolarizationLabel


class TestPolarizationExpansionAndContraction(unittest.TestCase):
    """
    Test expansion and contraction of the Polarization class
    """

    def test_all_cases(self) -> None:
        """
        Root test, runs the cases and validates them
        """
        test_cases = []
        test_cases.append(
            (Polarization(), PolarizationLabel.H, [[1], [0]], [[1, 0], [0, 0]])
        )
        test_cases.append(
            (
                Polarization(PolarizationLabel.H),
                PolarizationLabel.H,
                [[1], [0]],
                [[1, 0], [0, 0]],
            )
        )
        test_cases.append(
            (
                Polarization(PolarizationLabel.V),
                PolarizationLabel.V,
                [[0], [1]],
                [[0, 0], [0, 1]],
            )
        )
        test_cases.append(
            (
                Polarization(PolarizationLabel.R),
                PolarizationLabel.R,
                [[1 / jnp.sqrt(2)], [1j / jnp.sqrt(2)]],
                [[1 / 2, -1j / 2], [1j / 2, 1 / 2]],
            )
        )
        test_cases.append(
            (
                Polarization(PolarizationLabel.L),
                PolarizationLabel.L,
                [[1 / jnp.sqrt(2)], [-1j / jnp.sqrt(2)]],
                [[1 / 2, 1j / 2], [-1j / 2, 1 / 2]],
            )
        )

        for t in test_cases:
            self.run_test(*t)

    def run_test(self, *t) -> None:
        pol = t[0]
        label = t[1]
        state_vector = jnp.array(t[2])
        density_matrix = jnp.array(t[3])

        self.initalization_test(pol, label)
        self.first_expansion_test(pol, state_vector)
        self.second_expansion_test(pol, density_matrix)
        # Third expansion should not do anything
        self.third_expansion_test(pol, density_matrix)

        # Test contractions
        self.first_contract_test(pol, state_vector)
        self.second_contract_test(pol, label)

    def initalization_test(self, pol: Polarization, label: PolarizationLabel) -> None:
        for item in [pol.index, pol.envelope, pol.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(pol.state == label)

    def first_expansion_test(self, pol: Polarization, state_vector: jnp.array) -> None:
        pol.expand()
        for item in [pol.index, pol.envelope, pol.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(state_vector, pol.state))

    def second_expansion_test(
        self, pol: Polarization, density_matrix: jnp.array
    ) -> None:
        pol.expand()
        for item in [pol.index, pol.envelope, pol.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(density_matrix, pol.state, atol=1e-04))

    def third_expansion_test(
        self, pol: Polarization, density_matrix: jnp.array
    ) -> None:
        pol.expand()
        for item in [pol.index, pol.envelope, pol.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(density_matrix, pol.state))

    def first_contract_test(self, pol: Polarization, state_vector: jnp.array) -> None:
        pol.contract(final=ExpansionLevel.Vector)
        for item in [pol.index, pol.envelope, pol.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(state_vector, pol.state))

    def second_contract_test(self, pol: Polarization, label: PolarizationLabel) -> None:
        pol.contract(final=ExpansionLevel.Label)
        for item in [pol.index, pol.envelope, pol.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(pol.state == label)

    def test_non_label_pure_state(self) -> None:
        pol = Polarization()
        pol.expand()
        state = jnp.array([[1 / jnp.sqrt(3)], [jnp.sqrt(2) / jnp.sqrt(3)]])
        pol.state = jnp.copy(state)

        pol.expand()
        dm = jnp.matmul(state, jnp.conjugate(state.T))
        for item in [pol.index, pol.envelope, pol.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(dm, pol.state))
        pol.contract()
        for item in [pol.index, pol.envelope, pol.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(state, pol.state))
        #
        pol.contract()
        for item in [pol.index, pol.envelope, pol.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(state, pol.state))


class TestPolarizationSmallFunctions(unittest.TestCase):
    """
    Test small methods within the Polarization Class
    """

    def test_extract(self) -> None:
        """
        Test if extract functionality works as intended
        """
        pol = Polarization()
        pol.extract(1)
        for item in [pol.state]:
            self.assertIsNone(item)
        self.assertEqual(pol.index, 1)

    def test_set_index(self) -> None:
        pol = Polarization()
        with self.assertRaises(ValueError) as context:
            pol.set_index(1)

        pol = Polarization()
        pol.extract(1)
        for item in [pol.state]:
            self.assertIsNone(item)
        self.assertEqual(pol.index, 1)
        pol.set_index(3)
        for item in [pol.state]:
            self.assertIsNone(item)
        self.assertEqual(pol.index, 3)
        pol.set_index(2, 3)
        for item in [pol.state]:
            self.assertIsNone(item)
        self.assertEqual(pol.index, (3, 2))

    def test_repr(self) -> None:
        """
        Test the __repr__ function
        """
        pol = Polarization()
        self.assertEqual(pol.__repr__(), "|H⟩")
        pol = Polarization(PolarizationLabel.H)
        self.assertEqual(pol.__repr__(), "|H⟩")
        pol = Polarization(PolarizationLabel.V)
        self.assertEqual(pol.__repr__(), "|V⟩")
        pol = Polarization(PolarizationLabel.R)
        self.assertEqual(pol.__repr__(), "|R⟩")
        pol = Polarization(PolarizationLabel.L)
        self.assertEqual(pol.__repr__(), "|L⟩")
        pol = Polarization()
        pol.expand()
        representation = pol.__repr__()
        representation = representation.split("\n")
        self.assertEqual(representation[0], "⎡ +1.00 + 0.00j ⎤")
        self.assertEqual(representation[1], "⎣ +0.00 + 0.00j ⎦")
        pol.expand()
        representation = pol.__repr__()
        representation = representation.split("\n")
        self.assertEqual(representation[0], "⎡ +1.00 + 0.00j   +0.00 + 0.00j ⎤")
        self.assertEqual(representation[1], "⎣ +0.00 + 0.00j   +0.00 + 0.00j ⎦")
        pol = Polarization()
        pol.extract(1)
        representation = pol.__repr__()
        self.assertEqual(representation, str(pol.uid))


class TestPolarizationMeasurement(unittest.TestCase):
    """
    Test Various measurements
    """

    def test_measure_H_state_vector(self) -> None:
        pol = Polarization(PolarizationLabel.H)
        m = pol.measure()
        self.assertEqual(
            m[pol], 0, "Measurement outcome when measuring H must always be 0"
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

    def test_measure_H_density_matrix(self) -> None:
        pol = Polarization(PolarizationLabel.H)
        pol.expand()
        pol.expand()
        m = pol.measure()
        self.assertEqual(
            m[pol], 0, "Measurement outcome when measuring H must always be 0"
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

    def test_measure_V_state_vector(self) -> None:
        pol = Polarization(PolarizationLabel.V)
        m = pol.measure()
        self.assertEqual(
            m[pol], 1, "Measurement outcome when measuring H must always be 1"
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

    def test_measure_V_density_matrix(self) -> None:
        pol = Polarization(PolarizationLabel.V)
        pol.expand()
        pol.expand()
        m = pol.measure()
        self.assertEqual(
            m[pol], 1, "Measurement outcome when measuring H must always be 1"
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

    def test_measure_R_state_vector(self) -> None:
        C = Config()
        C.set_seed(1)
        pol = Polarization(PolarizationLabel.R)
        m = pol.measure()
        self.assertEqual(
            m[pol],
            0,
            "Measurement outcome when measuring R must always be 0, when seed is set to 1",
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

        C.set_seed(3)
        pol = Polarization(PolarizationLabel.R)
        pol.expand()
        m = pol.measure()
        self.assertEqual(
            m[pol],
            1,
            "Measurement outcome when measuring R must always be 1, when seed is set to 3",
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

    def test_measure_R_density_matrix(self) -> None:
        C = Config()
        C.set_seed(1)
        pol = Polarization(PolarizationLabel.R)
        pol.expand()
        pol.expand()
        m = pol.measure()
        self.assertEqual(
            m[pol],
            0,
            "Measurement outcome when measuring R must always be 0, when seed is set to 1",
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

        C.set_seed(3)
        pol = Polarization(PolarizationLabel.R)
        pol.expand()
        pol.expand()
        m = pol.measure()
        self.assertEqual(
            m[pol],
            1,
            "Measurement outcome when measuring R must always be 1, when seed is set to 3",
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

    def test_measure_L_state_vector(self) -> None:
        C = Config()
        C.set_seed(1)
        pol = Polarization(PolarizationLabel.L)
        m = pol.measure()
        self.assertEqual(
            m[pol],
            0,
            "Measurement outcome when measuring L must always be 0, when seed is set to 1",
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

        C.set_seed(3)
        pol = Polarization(PolarizationLabel.R)
        pol.expand()
        m = pol.measure()
        self.assertEqual(
            m[pol],
            1,
            "Measurement outcome when measuring R must always be 1, when seed is set to 3",
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

    def test_measure_L_density_matrix(self) -> None:
        C = Config()
        C.set_seed(1)
        pol = Polarization(PolarizationLabel.L)
        pol.expand()
        pol.expand()
        m = pol.measure()
        self.assertEqual(
            m[pol],
            0,
            "Measurement outcome when measuring L must always be 0, when seed is set to 1",
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

        C.set_seed(3)
        pol = Polarization(PolarizationLabel.R)
        pol.expand()
        m = pol.measure()
        self.assertEqual(
            m[pol],
            1,
            "Measurement outcome when measuring R must always be 1, when seed is set to 3",
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

    def test_POVM_measurement_state_vector(self) -> None:
        C = Config()
        C.set_seed(1)
        pol = Polarization()
        povm_operators = []
        povm_operators.append(jnp.array([[1, 0], [0, 0]]))
        povm_operators.append(jnp.array([[0, 0], [0, 1]]))
        m = pol.measure_POVM(povm_operators)
        self.assertEqual(
            m[0], 0, "Measurement outcome when measuring H must always be 0"
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)

    def test_POVM_measurement_density_matrix(self) -> None:
        C = Config()
        C.set_seed(1)
        pol = Polarization()
        pol.expand()
        pol.expand()
        povm_operators = []
        povm_operators.append(jnp.array([[1, 0], [0, 0]]))
        povm_operators.append(jnp.array([[0, 0], [0, 1]]))
        m = pol.measure_POVM(povm_operators)
        self.assertEqual(
            m[0], 0, "Measurement outcome when measuring H must always be 0"
        )
        self.assertTrue(
            pol.measured, "Polarization must have measurement=True after measurement."
        )
        for item in [
            pol.envelope,
            pol.composite_envelope,
            pol.state,
            pol.expansion_level,
        ]:
            self.assertIsNone(item)


class TestPolarizationKrausOperatorApplication(unittest.TestCase):
    def test_kraus_operators(self):
        C = Config()
        C.set_contraction(True)
        pol = Polarization()
        operators = [jnp.array([[1, 0], [0, 0]]), jnp.array([[0, 0], [0, 1]])]
        pol.apply_kraus(operators)
        self.assertEqual(pol.state, PolarizationLabel.H)

        operators = [jnp.array([[1, 0], [0, 0]]), jnp.array([[0, 0], [1, 0]])]
        operators = [jnp.array([[1, 0], [0, 0]]), jnp.array([[0, 0], [1, 0]])]

        # Correct usage of assertRaises
        with self.assertRaises(ValueError):
            pol.apply_kraus(operators)

        operators = [
            jnp.array([[1, 0, 0], [0, 0, 0]]),
            jnp.array([[0, 0, 0], [1, 0, 0]]),
        ]

        # Correct usage of assertRaises
        with self.assertRaises(ValueError):
            pol.apply_kraus(operators)
