from photon_weave.operation.composite_operation import (
    CompositeOperation,
    CompositeOperationType,
)
from photon_weave.operation.fock_operation import FockOperation, FockOperationType
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.state.envelope import Envelope

env1 = Envelope()
op = FockOperation(operation=FockOperationType.Creation)
env1.apply_operation(op)
vac1 = Envelope()

c1 = CompositeEnvelope(env1, vac1)

bs1 = CompositeOperation(CompositeOperationType.NonPolarizingBeamSplit)

# First Beam Splitter
c1.apply_operation(bs1, env1, vac1)
