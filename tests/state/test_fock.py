import random
import unittest

import jax.numpy as jnp
import numpy as np
import pytest

from photon_weave.photon_weave import Config
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.state.expansion_levels import ExpansionLevel
from photon_weave.state.fock import Fock
from photon_weave.state.polarization import Polarization, PolarizationLabel


class TestFockSmallFunctions(unittest.TestCase):
    """
    Test small methods withing the Fock class
    """

    def test_repr(self) -> None:
        fock = Fock()
        # Test the Label representation
        for i in range(100):
            fock.state = i
            self.assertEqual(fock.__repr__(), f"|{i}⟩")

        # Test the state vector representation
        fock.dimensions = 5
        label = random.randint(0, 4)
        fock.state = label
        fock.expand()
        representation = fock.__repr__().split("\n")
        for ln, line in enumerate(representation):
            if ln == 0:
                if label == ln:
                    self.assertEqual("⎡ +1.00 + 0.00j ⎤", line)
                else:
                    self.assertEqual("⎡ +0.00 + 0.00j ⎤", line)
            elif ln == len(representation) - 1:
                if label == ln:
                    self.assertEqual("⎣ +1.00 + 0.00j ⎦", line)
                else:
                    self.assertEqual("⎣ +0.00 + 0.00j ⎦", line)
            else:
                if label == ln:
                    self.assertEqual("⎢ +1.00 + 0.00j ⎥", line)
                else:
                    self.assertEqual("⎢ +0.00 + 0.00j ⎥", line)
        fock.expand()
        representation = fock.__repr__().split("\n")
        v = lambda x: f" +{x}.00 + 0.00j "
        for ln, line in enumerate(representation):
            constructed_line_1 = " ".join(
                [v(1) if ln == i else v(0) for i in range(fock.dimensions)]
            )
            constructed_line_0 = " ".join([v(0) for i in range(fock.dimensions)])
            if ln == 0:
                if label == ln:
                    self.assertEqual(f"⎡{constructed_line_1}⎤", line)
                else:
                    self.assertEqual(f"⎡{constructed_line_0}⎤", line)
            elif ln == len(representation) - 1:
                if label == ln:
                    self.assertEqual(f"⎣{constructed_line_1}⎦", line)
                else:
                    self.assertEqual(f"⎣{constructed_line_0}⎦", line)
            else:
                if label == ln:
                    self.assertEqual(f"⎢{constructed_line_1}⎥", line)
                else:
                    self.assertEqual(f"⎢{constructed_line_0}⎥", line)

    def test_equality(self) -> None:
        f1 = Fock()
        f2 = Fock()
        pol = Polarization()
        self.assertTrue(f1 == f2)
        self.assertFalse(f1 == pol)
        f1.expand()
        self.assertFalse(f1 == f2)
        f2.expand()
        self.assertTrue(f1 == f2)
        f1.expand()
        self.assertFalse(f1 == f2)
        f2.expand()
        self.assertTrue(f1 == f2)
        f1 = Fock()
        f1.state = 1
        f2 = Fock()
        self.assertTrue(f1 != f2)
        f1.expand()
        f2.expand()
        self.assertTrue(f1 != f2)
        f1.expand()
        f2.expand()
        self.assertTrue(f1 != f2)

    def test_extract(self) -> None:
        fock = Fock()
        fock.extract(1)
        for item in [fock.state]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, 1)
        fock = Fock()
        fock.extract((1, 1))
        for item in [fock.state]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, (1, 1))

        fock = Fock()
        fock.expand()
        fock.extract(1)
        for item in [fock.state]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, 1)
        fock = Fock()
        fock.expand()
        fock.extract((1, 1))
        for item in [fock.state]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, (1, 1))

        fock = Fock()
        fock.expand()
        fock.expand()
        fock.extract(1)
        for item in [fock.state]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, 1)
        fock = Fock()
        fock.expand()
        fock.expand()
        fock.extract((1, 1))
        for item in [fock.state]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, (1, 1))

    def test_set_index(self) -> None:
        fock = Fock()
        fock.set_index(1)
        self.assertEqual(fock.index, 1)
        fock.set_index(1, 1)
        self.assertEqual(fock.index, (1, 1))

    def test_set_measured(self) -> None:
        fock = Fock()
        fock._set_measured()
        self.assertTrue(fock.measured)
        for item in [fock.state]:
            self.assertIsNone(item)

        fock = Fock()
        fock.expand()
        fock._set_measured()
        self.assertTrue(fock.measured)
        for item in [fock.index, fock.state]:
            self.assertIsNone(item)

        fock = Fock()
        fock.expand()
        fock.expand()
        fock._set_measured()
        self.assertTrue(fock.measured)
        for item in [fock.index, fock.state]:
            self.assertIsNone(item)

    def test_num_quanta(self) -> None:
        fock = Fock()
        self.assertEqual(fock._num_quanta, 0)
        fock.expand()
        self.assertEqual(fock._num_quanta, 0)
        fock.expand()
        self.assertEqual(fock._num_quanta, 0)


class TestFockExpansionAndContraction(unittest.TestCase):
    def test_all_cases(self) -> None:
        """
        Root test
        """
        test_cases = []
        test_cases.append(
            (
                Fock(),
                0,
                [[1], [0], [0], [0], [0]],
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            )
        )

        test_cases.append(
            (
                Fock(),
                1,
                [[0], [1], [0], [0], [0]],
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            )
        )
        test_cases.append(
            (
                Fock(),
                2,
                [[0], [0], [1], [0], [0]],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            )
        )
        test_cases.append(
            (
                Fock(),
                3,
                [[0], [0], [0], [1], [0]],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                ],
            )
        )
        test_cases.append(
            (
                Fock(),
                4,
                [[0], [0], [0], [0], [1]],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                ],
            )
        )
        for tc in test_cases:
            tc[0].dimensions = 5
            tc[0].label = tc[1]

        for tc in test_cases:
            self.run_test(*tc)

    def run_test(self, *tc):
        fock = tc[0]
        label = tc[1]
        fock.state = label
        state_vector = jnp.array(tc[2])
        density_matrix = jnp.array(tc[3])
        self.initialization_test(fock, label)
        self.first_expansion_test(fock, state_vector)
        self.second_expansion_test(fock, density_matrix)
        self.third_expansion_test(fock, density_matrix)
        self.first_contract_test(fock, state_vector)
        self.second_contract_test(fock, label)
        self.third_contract_test(fock, label)

    def initialization_test(self, fock: Fock, label: int) -> None:
        for i, item in enumerate([fock.envelope, fock.composite_envelope, fock.index]):
            self.assertIsNone(item, f"{i}-{item}")
        self.assertEqual(fock.state, label)

    def first_expansion_test(self, fock: Fock, state_vector: jnp.ndarray) -> None:
        fock.expand()
        for item in [fock.index, fock.envelope, fock.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(state_vector, fock.state))

    def second_expansion_test(self, fock: Fock, density_matrix: jnp.ndarray) -> None:
        fock.expand()
        for item in [fock.index, fock.envelope, fock.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(density_matrix, fock.state))

    def third_expansion_test(self, fock: Fock, density_matrix: jnp.ndarray) -> None:
        fock.expand()
        for item in [fock.index, fock.envelope, fock.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(density_matrix, fock.state))

    def first_contract_test(self, fock: Fock, state_vector: jnp.ndarray) -> None:
        fock.contract(final=ExpansionLevel.Vector)
        for item in [fock.index, fock.envelope, fock.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(state_vector, fock.state))

    def second_contract_test(self, fock: Fock, label: int) -> None:
        fock.contract(final=ExpansionLevel.Label)
        for item in [fock.index, fock.envelope, fock.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(label, fock.state))

    def third_contract_test(self, fock: Fock, label: int) -> None:
        fock.contract(final=ExpansionLevel.Label)
        for item in [fock.index, fock.envelope, fock.composite_envelope]:
            self.assertIsNone(item)
        self.assertTrue(jnp.allclose(label, fock.state))


class TestFockMeasurement(unittest.TestCase):
    """
    Test Various Measurements
    """

    def test_general_measurement_label(self) -> None:
        for i in range(10):
            f = Fock()
            f.state = i
            m = f.measure()
            self.assertEqual(m[f], i)

    def test_general_measurement_vector(self) -> None:
        for i in range(10):
            f = Fock()
            f.state = i
            f.expand()
            m = f.measure()
            self.assertEqual(m[f], i)

    def test_general_measurement_matrix(self) -> None:
        for i in range(10):
            f = Fock()
            f.state = i
            f.expand()
            f.expand()
            m = f.measure()
            self.assertEqual(m[f], i)

    def test_superposition_vector(self) -> None:
        f = Fock()
        f.dimensions = 2
        f.expand()
        f.state = jnp.array([[1 / jnp.sqrt(2)], [1 / jnp.sqrt(2)]])
        C = Config()
        C.set_seed(1)
        m = f.measure()
        self.assertEqual(m[f], 1, "Should be 1 with seed 1")
        f = Fock()
        f.dimensions = 2
        f.expand()
        f.state = jnp.array([[1 / jnp.sqrt(2)], [1 / jnp.sqrt(2)]])
        C.set_seed(3)
        m = f.measure()
        self.assertEqual(m[f], 1, "Should be 1 with seed 3")

    def test_superposition_matrix(self) -> None:
        f = Fock()
        f.dimensions = 2
        f.expand()
        f.state = jnp.array([[1 / jnp.sqrt(2)], [1 / jnp.sqrt(2)]])
        f.expand()
        C = Config()
        C.set_seed(1)
        m = f.measure()
        self.assertEqual(m[f], 1, "Should be 1 with seed 1")
        f = Fock()
        f.dimensions = 2
        f.expand()
        f.state = jnp.array([[1 / jnp.sqrt(2)], [1 / jnp.sqrt(2)]])
        f.expand()
        C.set_seed(3)
        m = f.measure()
        self.assertEqual(m[f], 1, "Should be 1 with seed 3")

    def test_POVM_measurement_state_vector(self) -> None:
        C = Config()
        C.set_seed(1)
        f = Fock()
        f.dimensions = 2
        f.expand()
        povm_operators = []
        povm_operators.append(jnp.array([[1, 0], [0, 0]]))
        povm_operators.append(jnp.array([[0, 0], [0, 1]]))
        m = f.measure_POVM(povm_operators)
        self.assertEqual(m, 0, "Measurement outcome when measuring H must always be 0")
        self.assertTrue(f.measured)
        for item in [f.label, f.expansion_level, f.state_vector, f.density_matrix]:
            self.assertIsNone(item)

    def test_POVM_measurement_state_vector(self) -> None:
        C = Config()
        C.set_seed(1)
        f = Fock()
        f.dimensions = 2
        povm_operators = []
        povm_operators.append(jnp.array([[1, 0], [0, 0]]))
        povm_operators.append(jnp.array([[0, 0], [0, 1]]))
        m = f.measure_POVM(povm_operators)
        self.assertEqual(
            m[0], 0, "Measurement outcome when measuring H must always be 0"
        )
        self.assertTrue(f.measured)
        for item in [f.index, f.state]:
            self.assertIsNone(item)


class TestFockDimensionChange(unittest.TestCase):
    def test_fock_dimension_change(self) -> None:
        f = Fock()
        f.resize(4)
        self.assertEqual(f.dimensions, 4)
        self.assertEqual(f.state, 0)

        f.expand()
        f.resize(5)
        self.assertEqual(f.dimensions, 5)
        self.assertTrue(jnp.allclose(f.state, jnp.array([[1], [0], [0], [0], [0]])))

        f.resize(2)
        self.assertEqual(f.dimensions, 2)
        self.assertTrue(jnp.allclose(f.state, jnp.array([[1], [0]])))

        f.expand()
        f.resize(4)
        self.assertEqual(f.dimensions, 4)
        self.assertTrue(
            jnp.allclose(
                f.state,
                jnp.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
            )
        )

        f.resize(2)
        self.assertEqual(f.dimensions, 2)
        self.assertTrue(jnp.allclose(f.state, jnp.array([[1, 0], [0, 0]])))

    def test_resize_in_envelope_vector(self) -> None:
        env = Envelope()
        env.polarization.state = PolarizationLabel.V
        env.fock.dimensions = 2
        env.combine()
        env.resize_fock(3)
        self.assertTrue(
            jnp.allclose(env.state, jnp.array([[0], [1], [0], [0], [0], [0]]))
        )
        self.assertEqual(env.fock.dimensions, 3)

        s = env.resize_fock(2)
        self.assertTrue(s)
        self.assertTrue(jnp.allclose(env.state, jnp.array([[0], [1], [0], [0]])))
        self.assertEqual(env.fock.dimensions, 2)

        # Trying the same with the reversed order
        env = Envelope()
        env.polarization.state = PolarizationLabel.R
        env.fock.state = 2
        env.fock.dimensions = 3
        env.combine()
        env.reorder(env.polarization, env.fock)

        s = env.resize_fock(2)
        self.assertFalse(s)
        self.assertEqual(env.fock.dimensions, 3)

        s = env.resize_fock(4)
        self.assertTrue(s)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [
                        [0],
                        [0],
                        [0],
                        [0],
                        [1 / jnp.sqrt(2)],
                        [1j / jnp.sqrt(2)],
                        [0],
                        [0],
                    ]
                ),
            )
        )

        s = env.resize_fock(2)
        self.assertFalse(s)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [
                        [0],
                        [0],
                        [0],
                        [0],
                        [1 / jnp.sqrt(2)],
                        [1j / jnp.sqrt(2)],
                        [0],
                        [0],
                    ]
                ),
            )
        )

        s = env.resize_fock(3)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array([[0], [0], [0], [0], [1 / jnp.sqrt(2)], [1j / jnp.sqrt(2)]]),
            )
        )

    def test_resize_in_envelope_matrix(self) -> None:
        env = Envelope()
        env.fock.dimensions = 2
        env.fock.state = 1
        env.polarization.state = PolarizationLabel.R
        env.combine()
        env.expand()
        s = env.resize_fock(3)
        self.assertTrue(s)
        self.assertEqual(env.fock.dimensions, 3)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0.5, -0.5j, 0, 0],
                        [0, 0, 0.5j, 0.5, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            )
        )
        s = env.resize_fock(1)
        self.assertFalse(s)
        self.assertEqual(env.fock.dimensions, 3)

        s = env.resize_fock(2)
        self.assertTrue(s)
        self.assertTrue(
            jnp.allclose(
                env.state,
                jnp.array(
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, -0.5j], [0, 0, 0.5j, 0.5]]
                ),
            )
        )

    def test_resize_in_composite_envelope_vector(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env1.fock.state = 1
        env2.polarization.state = PolarizationLabel.R

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.polarization)
        s = env1.fock.resize(3)
        self.assertTrue(s)
        self.assertEqual(env1.fock.dimensions, 3)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array([[0], [0], [1 / jnp.sqrt(2)], [1j / jnp.sqrt(2)], [0], [0]]),
            )
        )

        s = ce.resize_fock(1, env1.fock)
        self.assertFalse(s)

        s = env1.resize_fock(2)
        self.assertTrue(s)
        self.assertEqual(env1.fock.dimensions, 2)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array([[0], [0], [1 / jnp.sqrt(2)], [1j / jnp.sqrt(2)]]),
            )
        )

    def test_resize_in_composite_envelope_matrix(self) -> None:
        env1 = Envelope()
        env2 = Envelope()
        env1.fock.dimensions = 2
        env1.fock.state = 1
        env2.polarization.state = PolarizationLabel.R

        ce = CompositeEnvelope(env1, env2)
        ce.combine(env1.fock, env2.polarization)
        ce.expand(env1.fock)
        s = ce.resize_fock(3, env1.fock)
        self.assertTrue(s)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0.5, -0.5j, 0, 0],
                        [0, 0, 0.5j, 0.5, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            )
        )
        self.assertEqual(env1.fock.dimensions, 3)

        s = env1.fock.resize(1)
        self.assertFalse(s)

        s = env1.fock.resize(2)
        self.assertTrue(s)
        self.assertTrue(
            jnp.allclose(
                ce.product_states[0].state,
                jnp.array(
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0.5, -0.5j], [0, 0, 0.5j, 0.5]]
                ),
            )
        )
        self.assertEqual(env1.fock.dimensions, 2)
