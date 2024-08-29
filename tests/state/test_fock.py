import unittest

import numpy as np
import jax.numpy as jnp
import random

from photon_weave.photon_weave import Config
from photon_weave.state.fock import Fock, FockAlreadyMeasuredException
from photon_weave.state.polarization import Polarization
from photon_weave.state.envelope import Envelope
from photon_weave.state.expansion_levels import ExpansionLevel

class TestFockSmallFunctions(unittest.TestCase):
    """
    Test small methods withing the Fock class
    """
    def test_repr(self) -> None:
        fock = Fock()
        # Test the Label representation
        for i in range(100):
           fock.label = i 
           self.assertEqual(fock.__repr__(), f"|{i}⟩")

        # Test the state vector representation
        fock.dimensions = 5
        label = random.randint(0,4)
        fock.label = label
        fock.expand()
        representation = fock.__repr__().split("\n")
        for ln, line in enumerate(representation):
            if ln == 0:
                if label == ln:
                    self.assertEqual("⎡ 1.00 + 0.00j ⎤", line)
                else:
                    self.assertEqual("⎡ 0.00 + 0.00j ⎤", line)
            elif ln == len(representation) -1:
                if label == ln:
                    self.assertEqual("⎣ 1.00 + 0.00j ⎦", line)
                else:
                    self.assertEqual("⎣ 0.00 + 0.00j ⎦", line)
            else:
                if label == ln:
                    self.assertEqual("⎢ 1.00 + 0.00j ⎥", line)
                else:
                    self.assertEqual("⎢ 0.00 + 0.00j ⎥", line)
        fock.expand()
        representation = fock.__repr__().split("\n")
        v = lambda x: f" {x}.00 + 0.00j "
        for ln, line in enumerate(representation):
            constructed_line_1 = " ".join([
                v(1) if ln == i else v(0) for i in range(fock.dimensions)
            ])
            constructed_line_0 = " ".join([
                v(0) for i in range(fock.dimensions)
            ])
            if ln == 0:
                if label == ln:
                    self.assertEqual(f"⎡{constructed_line_1}⎤", line)
                else:
                    self.assertEqual(f"⎡{constructed_line_0}⎤", line)
            elif ln == len(representation) -1:
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
        f1.label = 1
        f2 = Fock()
        self.assertTrue(f1 != f2)
        f1.expand()
        f2.expand()
        self.assertTrue(f1 != f2)
        f1.expand()
        f2.expand()
        self.assertTrue(f1 != f2)

    def test_extract(self) -> None:
        fock= Fock()
        fock.extract(1)
        for item in [fock.label, fock.state_vector, fock.density_matrix]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, 1)
        fock= Fock()
        fock.extract((1,1))
        for item in [fock.label, fock.state_vector, fock.density_matrix]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, (1,1))

        fock= Fock()
        fock.expand()
        fock.extract(1)
        for item in [fock.label, fock.state_vector, fock.density_matrix]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, 1)
        fock= Fock()
        fock.expand()
        fock.extract((1,1))
        for item in [fock.label, fock.state_vector, fock.density_matrix]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, (1,1))

        fock= Fock()
        fock.expand()
        fock.expand()
        fock.extract(1)
        for item in [fock.label, fock.state_vector, fock.density_matrix]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, 1)
        fock= Fock()
        fock.expand()
        fock.expand()
        fock.extract((1,1))
        for item in [fock.label, fock.state_vector, fock.density_matrix]:
            self.assertIsNone(item)
        self.assertEqual(fock.index, (1,1))

    def test_set_index(self) -> None:
        fock = Fock()
        fock.set_index(1)
        self.assertEqual(fock.index, 1)
        fock.set_index(1,1)
        self.assertEqual(fock.index, (1,1))

    def test_normalization(self) -> None:
        fock = Fock()
        fock.expand()
        fock.state_vector[0][0] = 2
        fock.normalize()
        self.assertEqual(fock.state_vector[0][0], 1)
        fock = Fock()
        fock.expand()
        fock.expand()
        fock.density_matrix[0][0] = 2
        fock.normalize()
        self.assertEqual(fock.density_matrix[0][0], 1)

    def test_set_measured(self) -> None:
        fock = Fock()
        fock._set_measured()
        self.assertTrue(fock.measured)
        for item in [fock.label, fock.expansion_level, fock.state_vector,
                     fock.density_matrix, fock.index]:
            self.assertIsNone(item)

        fock = Fock()
        fock.expand()
        fock._set_measured()
        self.assertTrue(fock.measured)
        for item in [fock.label, fock.expansion_level, fock.state_vector,
                     fock.density_matrix, fock.index]:
            self.assertIsNone(item)
            
        fock = Fock()
        fock.expand()
        fock.expand()
        fock._set_measured()
        self.assertTrue(fock.measured)
        for item in [fock.label, fock.expansion_level, fock.state_vector,
                     fock.density_matrix, fock.index]:
            self.assertIsNone(item)

    def test_num_quanta(self) -> None:
        fock = Fock()
        self.assertEqual(fock._num_quanta, 0)
        fock.expand()
        self.assertEqual(fock._num_quanta, 0)
        fock.expand()
        self.assertEqual(fock._num_quanta, 0)

    def test_get_subspace(self) -> None:
        f = Fock()
        self.assertEqual(f.get_subspace(), 0)
        f.dimensions = 2
        f.expand()
        self.assertTrue(
            jnp.allclose(
                f.state_vector,
                jnp.array(
                    [[1],[0]]
                )
            )
        )
        f.expand()
        self.assertTrue(
            jnp.allclose(
                f.density_matrix,
                jnp.array(
                    [[1,0],[0,0]]
                )
            )
        )

class TestFockExpansionAndContraction(unittest.TestCase):
    def test_all_cases(self) -> None:
        """
        Root test
        """
        test_cases = []
        test_cases.append(
            (Fock(), 0, [[1],[0],[0],[0],[0]],
             [
                 [1,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0]
             ]))

        test_cases.append(
            (Fock(), 1, [[0],[1],[0],[0],[0]],
             [
                 [0,0,0,0,0],
                 [0,1,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0]
             ])
        )
        test_cases.append(
            (Fock(), 2, [[0],[0],[1],[0],[0]],
             [
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,1,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0]
             ])
        )
        test_cases.append(
            (Fock(), 3, [[0],[0],[0],[1],[0]],
             [
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,1,0],
                 [0,0,0,0,0]
             ])
        )
        test_cases.append(
            (Fock(), 4, [[0],[0],[0],[0],[1]],
             [
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,1]
             ])
        )
        for tc in test_cases:
            tc[0].dimensions = 5
            tc[0].label = tc[1]

        for tc in test_cases:
            self.run_test(*tc)

    def run_test(self, *tc):
        fock = tc[0]
        label = tc[1]
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
        for i,item in enumerate([fock.index, fock.state_vector, fock.density_matrix, fock.envelope]):
            self.assertIsNone(item)
        self.assertEqual(fock.label, label)

    def first_expansion_test(self, fock: Fock, state_vector: jnp.ndarray) -> None:
        fock.expand()
        for item in [fock.index, fock.label, fock.density_matrix, fock.envelope]:
            self.assertIsNone(item)
        self.assertTrue(
            jnp.allclose(
                state_vector,
                fock.state_vector
            )
        )

    def second_expansion_test(self, fock: Fock, density_matrix: jnp.ndarray) -> None:
        fock.expand()
        for item in [fock.index, fock.label, fock.state_vector, fock.envelope]:
            self.assertIsNone(item)
        self.assertTrue(
            jnp.allclose(
                density_matrix,
                fock.density_matrix
            )
        )

    def third_expansion_test(self, fock: Fock, density_matrix: jnp.ndarray) -> None:
        fock.expand()
        for item in [fock.index, fock.label, fock.state_vector, fock.envelope]:
            self.assertIsNone(item)
        self.assertTrue(
            jnp.allclose(
                density_matrix,
                fock.density_matrix
            )
        )

    def first_contract_test(self, fock:Fock, state_vector: jnp.ndarray) -> None:
        fock.contract(final=ExpansionLevel.Vector)
        for item in [fock.index, fock.label, fock.density_matrix, fock.envelope]:
            self.assertIsNone(item)
        self.assertTrue(
            jnp.allclose(
                state_vector,
                fock.state_vector
            )
        )

    def second_contract_test(self, fock:Fock, label: int) -> None:
        fock.contract(final=ExpansionLevel.Label)
        for item in [fock.index, fock.state_vector, fock.density_matrix, fock.envelope]:
            self.assertIsNone(item)
        self.assertTrue(
            jnp.allclose(
                label,
                fock.label
            )
        )

    def third_contract_test(self, fock:Fock, label: int) -> None:
        fock.contract(final=ExpansionLevel.Label)
        for item in [fock.index, fock.state_vector, fock.density_matrix, fock.envelope]:
            self.assertIsNone(item)
        self.assertTrue(
            jnp.allclose(
                label,
                fock.label
            )
        )


class TestFockMeasurement(unittest.TestCase):
    """
    Test Various Measurements
    """
    def test_general_measurement_label(self) -> None:
        for i in range(10):
            f = Fock()
            f.label = i
            m = f.measure()
            self.assertEqual(m, i)

    def test_general_measurement_vector(self) -> None:
        for i in range(10):
            f = Fock()
            f.label = i
            f.expand()
            m = f.measure()
            self.assertEqual(m, i)

    def test_general_measurement_matrix(self) -> None:
        for i in range(10):
            f = Fock()
            f.label = i
            f.expand()
            f.expand()
            m = f.measure()
            self.assertEqual(m, i)

    def test_superposition_vector(self) -> None:
        f = Fock()
        f.dimensions = 2
        f.expand()
        f.state_vector = jnp.array(
            [[1/jnp.sqrt(2)],[1/jnp.sqrt(2)]]
        )
        C = Config()
        C.set_seed(1)
        m = f.measure()
        self.assertEqual(m, 1, "Should be 1 with seed 1")
        f = Fock()
        f.dimensions = 2
        f.expand()
        f.state_vector = jnp.array(
            [[1/jnp.sqrt(2)],[1/jnp.sqrt(2)]]
        )
        C.set_seed(3)
        m = f.measure()
        self.assertEqual(m, 0, "Should be 1 with seed 3")

    def test_superposition_matrix(self) -> None:
        f = Fock()
        f.dimensions = 2
        f.expand()
        f.state_vector = jnp.array(
            [[1/jnp.sqrt(2)],[1/jnp.sqrt(2)]]
        )
        f.expand()
        C = Config()
        C.set_seed(1)
        m = f.measure()
        self.assertEqual(m, 1, "Should be 1 with seed 1")
        f = Fock()
        f.dimensions = 2
        f.expand()
        f.state_vector = jnp.array(
            [[1/jnp.sqrt(2)],[1/jnp.sqrt(2)]]
        )
        f.expand()
        C.set_seed(3)
        m = f.measure()
        self.assertEqual(m, 0, "Should be 1 with seed 3")

    def test_double_measurement(self) -> None:
        f = Fock()
        f.measure()

        with self.assertRaises(FockAlreadyMeasuredException) as context:
            f.measure()

    def test_POVM_measurement_state_vector(self) -> None:
        C = Config()
        C.set_seed(1)
        f = Fock()
        f.dimensions = 2
        f.expand()
        povm_operators = []
        povm_operators.append(jnp.array([[1,0],[0,0]]))
        povm_operators.append(jnp.array([[0,0],[0,1]]))
        m = f.measure_POVM(povm_operators)
        self.assertEqual(m,0, "Measurement outcome when measuring H must always be 0")
        self.assertTrue(f.measured)
        for item in [f.label, f.expansion_level, f.state_vector, f.density_matrix]:
            self.assertIsNone(item)

    def test_POVM_measurement_state_vector(self) -> None:
        C = Config()
        C.set_seed(1)
        f = Fock()
        f.dimensions = 2
        povm_operators = []
        povm_operators.append(jnp.array([[1,0],[0,0]]))
        povm_operators.append(jnp.array([[0,0],[0,1]]))
        m = f.measure_POVM(povm_operators)
        self.assertEqual(m,0, "Measurement outcome when measuring H must always be 0")
        self.assertTrue(f.measured)
        for item in [f.label, f.expansion_level, f.state_vector, f.density_matrix]:
            self.assertIsNone(item)


class TestFockResizse(unittest.TestCase):
    def test_resizse_label(self):
        f = Fock()
        f.dimensions = 3
        success = f.resize(5)
        self.assertEqual(f.dimensions, 5)
        self.assertTrue(success)
        self.assertEqual(f.label, 0)
        self.assertEqual(f.expansion_level, ExpansionLevel.Label)
        for item in [f.state_vector, f.density_matrix]:
            self.assertIsNone(item)

        f = Fock()
        f.dimensions = 5
        success = f.resize(3)
        self.assertEqual(f.dimensions, 3)
        self.assertTrue(success)
        self.assertEqual(f.expansion_level, ExpansionLevel.Label)
        self.assertEqual(f.label, 0)
        for item in [f.state_vector, f.density_matrix]:
            self.assertIsNone(item)

    def test_resizse_vector(self):
        f = Fock()
        f.dimensions = 3
        f.expand()
        success = f.resize(5)
        self.assertEqual(f.dimensions, 5)
        self.assertTrue(success)
        self.assertEqual(f.expansion_level, ExpansionLevel.Vector)
        self.assertTrue(
            jnp.allclose(
                f.state_vector,
                jnp.array(
                    [[1],[0],[0],[0],[0]]
                )
            )
        )
        for item in [f.label, f.density_matrix]:
            self.assertIsNone(item)

        f = Fock()
        f.dimensions = 5
        f.expand()
        success = f.resize(3)
        self.assertEqual(f.dimensions, 3)
        self.assertTrue(success)
        self.assertEqual(f.expansion_level, ExpansionLevel.Vector)
        self.assertTrue(
            jnp.allclose(
                f.state_vector,
                jnp.array(
                    [[1],[0],[0]]
                )
            )
        )
        for item in [f.label, f.density_matrix]:
            self.assertIsNone(item)

    def test_resizse_matrix(self):
        f = Fock()
        f.dimensions = 3
        f.expand()
        f.expand()
        success = f.resize(5)
        self.assertEqual(f.dimensions, 5)
        self.assertTrue(success)
        self.assertEqual(f.expansion_level, ExpansionLevel.Matrix)
        self.assertTrue(
            jnp.allclose(
                f.density_matrix,
                jnp.array(
                    [[1,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0],
                     [0,0,0,0,0]]
                )
            )
        )
        for item in [f.label, f.state_vector]:
            self.assertIsNone(item)

        f = Fock()
        f.dimensions = 5
        f.expand()
        f.expand()
        success = f.resize(3)
        self.assertEqual(f.dimensions, 3)
        self.assertTrue(success)
        self.assertEqual(f.expansion_level, ExpansionLevel.Matrix)
        self.assertTrue(
            jnp.allclose(
                f.density_matrix,
                jnp.array(
                    [[1,0,0],
                     [0,0,0],
                     [0,0,0]]
                )
            )
        )
        for item in [f.label, f.state_vector]:
            self.assertIsNone(item)
