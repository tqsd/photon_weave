import unittest
import jax.numpy as jnp

from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope
from photon_weave.operation import Operation, CompositeOperationType
from photon_weave.state.utils.measurements import (
    measure_vector, measure_matrix
)


class TestMeasurementUtils(unittest.TestCase):
    def test_measurement_vector(self) -> None:
        env1 = Envelope()
        env1.fock.state= 1
        env2 = Envelope()
        env3 = Envelope()
        env4 = Envelope()

        measurement_probabilities = {}

        def log_probabilities(fock, probabilities):
            measurement_probabilities[fock] = probabilities

        # create the state 
        op = Operation(
            CompositeOperationType.NonPolarizingBeamSplitter,
            eta=jnp.pi/4)

        ce = CompositeEnvelope(env1, env2, env3, env4)
        ce.apply_operation(op, env1.fock, env2.fock)
        ce.apply_operation(op, env1.fock, env3.fock)
        ce.apply_operation(op, env4.fock, env2.fock)

        ce.reorder(env1.fock, env2.fock, env3.fock, env4.fock)

        for env in [env1, env2, env3, env4]:
            measure_vector(
                [env1.fock, env2.fock, env3.fock, env4.fock],
                [env.fock],
                ce.states[0].state,
                prob_callback=log_probabilities
                )
        for state, item in measurement_probabilities.items():
            assert jnp.isclose(float(item[0]), 0.75, atol=1e-5)
            assert jnp.isclose(float(item[1]), 0.25, atol=1e-5)

    def test_measurement_matrix(self) -> None:
        env1 = Envelope()
        env1.fock.state= 1
        env2 = Envelope()
        env3 = Envelope()
        env4 = Envelope()

        measurement_probabilities = {}

        def log_probabilities(fock, probabilities):
            measurement_probabilities[fock] = probabilities

        # create the state 
        op = Operation(
            CompositeOperationType.NonPolarizingBeamSplitter,
            eta=jnp.pi/4)

        ce = CompositeEnvelope(env1, env2, env3, env4)
        ce.apply_operation(op, env1.fock, env2.fock)
        ce.apply_operation(op, env1.fock, env3.fock)
        ce.apply_operation(op, env4.fock, env2.fock)

        ce.reorder(env1.fock, env2.fock, env3.fock, env4.fock)

        env1.fock.expand()
        print(ce.states[0].state)

        for env in [env1, env2, env3, env4]:
            measure_matrix(
                [env1.fock, env2.fock, env3.fock, env4.fock],
                [env.fock],
                ce.states[0].state,
                prob_callback=log_probabilities
                )
        for state, item in measurement_probabilities.items():
            print(item)
            assert jnp.isclose(float(item[0]), 0.75, atol=1e-5)
            assert jnp.isclose(float(item[1]), 0.25, atol=1e-5)
