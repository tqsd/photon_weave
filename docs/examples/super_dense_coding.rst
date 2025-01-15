Super Dense Coding
==================

This tutorial showcases elementary super dense coding protocol, where the information is encoded into a polarization component of the pulse.

It is an exercise in showing how one can implement interoperable devices, which at the time of developing do not need to care about the construction of the product spaces, but can still operate on the spaces, even though they might be elements of a bigger product space.


In this tutorial we will implement individual components as a reusable python classes.

Entangled Photon Source
-----------------------

We start by constructing the entangled photon source. This class has one method `emit()`, which produces two `Envelope` objects, which are entangled in the polarization components.

.. math::
   
   \begin{align}
   |\Phi^+\rangle = \frac{1}{\sqrt{2}} ( |00\rangle + |11\rangle )
   \end{align}

.. code:: python

    from photon_weave.state.envelope import Envelope
    from photon_weave.state.composite_envelope import CompositeEnvelope
    from photon_weave.operation.polarization_operation import PolarizationOperationType
    from photon_weave.operation.operation import Operation
    from photon_weave.operation.composite_operation import CompositeOperationType
    
    class EntangledPhotonSource:
    
        def emit(self) -> tuple[Envelope, Envelope]:
            env1 = Envelope()
            env2 = Envelope()
            env1.polarization.expand()
            env2.polarization.expand()
    
            env1.fock.state = 1
            env2.fock.state = 1
    
            ce = CompositeEnvelope(env1, env2)
            ce.combine(env1.polarization, env2.polarization)
    
            # Entangle the polarizations
            op_h = Operation(PolarizationOperationType.H)
            op_cnot = Operation(CompositeOperationType.CXPolarization)
    
            ce.apply_operation(op_h, env1.polarization)
            ce.apply_operation(op_cnot, env1.polarization, env2.polarization)
    
            return env1, env2


Envelope Buffer
---------------
Super dense protocol works on the premise of preshared entanglement, thus, we must store the distributed entangled pairs. We will use Python built-in `Queue` in order to manage the storage in envelopes.

.. code:: python

    import queue
    from photon_weave.state.envelope import Envelope
    
    class EnvelopeBuffer:
        def __init__(self):
            self.buffer = queue.Queue()
    
        def store(self, env: Envelope):
            self.buffer.put(env)
    
        def get(self) -> Envelope:
            return self.buffer.get()    

Super Dense Encoder
-------------------
Once the entangled pairs are distributed we can use the part stored at the sender in order to encode two bits of information. To encode the two bits into the EPR pair the following operators are applied:

.. math::
   \begin{align}
   &(0,0) \to |\Phi^+\rangle = (\mathbb{I} \otimes \mathbb{I}) |\Phi^+\rangle\\
   &(0,1) \to |\Psi^+\rangle = (X \otimes \mathbb{I}) |\Phi^+\rangle\\
   &(1,0) \to |\Phi^-\rangle = (Z \otimes \mathbb{I}) |\Phi^+\rangle\\
   &(1,1) \to |\Psi^-\rangle = ((X \cdot Z) \otimes \mathbb{I}) |\Phi^+\rangle\\
   \end{align}
   

.. code:: python

    from photon_weave.state.envelope import Envelope
    from photon_weave.operation.operation import Operation
    from photon_weave.operation.polarization_operation import PolarizationOperationType
    
    
    class DenseEncoder:
        def encode(self, message: tuple[int, int], env: Envelope) -> Envelope:
            """
            Encode two bits into entangled photon
            """
            op_x = Operation(PolarizationOperationType.X)
            op_z = Operation(PolarizationOperationType.Z)
    
            ce = env.composite_envelope.states[0]
            match message:
                case (0, 0):
                    pass
                case (0, 1):
                    env.polarization.apply_operation(op_x)
                case (1, 0):
                    env.polarization.apply_operation(op_z)
                case (1, 1):
                    env.polarization.apply_operation(op_x)
                    env.polarization.apply_operation(op_z)
            
            return env


Super Dense Decoder
-------------------
Once the sender (Alice) encodes the classical bits \((a,b)\) by applying 
:math:`(X^a Z^b \otimes \mathbb{I})` to her half of the shared Bell state 
:math:`|\Phi^+\rangle`, she sends that qubit to the receiver (Bob). Bob then 
performs a Bell-state measurement on the two qubits. Mathematically, this can be 
expressed as follows:

.. math::
   \begin{aligned}
   %
   % 1) Alice's encoding on her qubit
   %
   (X^a Z^b \otimes \mathbb{I}) \, |\Phi^+\rangle
     &= \text{one of the four Bell states (}\Phi^\pm \text{, } \Psi^\pm\text{)}, \\[6pt]
   %
   % 2) Bob applies CNOT (qubit 1 -> qubit 2)
   %
   \mathrm{CNOT}_{1 \to 2} \bigl(X^a Z^b \otimes \mathbb{I}\bigr) \, |\Phi^+\rangle
     &= \text{(intermediate disentangled state)}, \\[6pt]
   %
   % 3) Bob applies Hadamard on qubit 1
   %
   (H \otimes \mathbb{I}) \,\mathrm{CNOT}_{1 \to 2} \bigl(X^a Z^b \otimes \mathbb{I}\bigr) \, |\Phi^+\rangle
     &= |a\,b\rangle, 
   \end{aligned}

where :math:`a,b \in \{0,1\}`. Finally, Bob measures both qubits in the 
computational basis, obtaining the two bits :math:`(a, b)` directly.

.. code:: python

    from photon_weave.state.envelope import Envelope
    from photon_weave.state.composite_envelope import CompositeEnvelope
    from photon_weave.operation import Operation
    from photon_weave.operation.composite_operation import CompositeOperationType
    from photon_weave.operation.polarization_operation import PolarizationOperationType
    
    class DenseDecoder:
    
        def decode(self, env1: Envelope, env2: Envelope) -> tuple[int, int]:
    
            op_h = Operation(PolarizationOperationType.H)
            op_cnot = Operation(CompositeOperationType.CXPolarization)
    
            ce = CompositeEnvelope(env1, env2)
            
            ce.apply_operation(op_cnot, env1.polarization, env2.polarization)
    
            env1.polarization.apply_operation(op_h)
            
            m1 = env1.measure()
            m2 = env2.measure()
    
            # Get the outcomes of the polarization measurements
            p1 = m1[env1.polarization]
            p2 = m2[env2.polarization]
    
            return p1, p2


Super Dense Protocol
--------------------

Finally, we can put the implemented components together into a working super dense protocol.

We start by importing all of the needed classes and modules:

.. code:: python
    from random import randint
    
    from photon_weave.state.envelope import Envelope
    
    from interoperable_devices import (
        DenseDecoder, DenseEncoder,
        EntangledPhotonSource, EnvelopeBuffer
        )

Then we build the sender class. The sender class will create the entangled pairs. It will store one envelope in its buffer and send the other half of the pair to the receiver. When sending the message, it will use `DenseEncoder` in order to encode the two bit message and then send its envelope to the receiver.

.. code:: python

    class DenseSender:
        def __init__(self):
            self.encoder = DenseEncoder()
            self.buffer = EnvelopeBuffer()
            self.source = EntangledPhotonSource()
            self.receiver = None
    
        def register_receiver(self, receiver: "DenseReceiver"):
            self.receiver = receiver
    
        def share_entanglement(self, pulses: int) -> None:
            for i in range(pulses):
                env1, env2 = self.source.emit()
                self.buffer.store(env1)
                self.receiver.receive_epr(env2)
    
        def send_message(self, message: tuple[int, int]) -> None:
            env = self.buffer.get()
            self.encoder.encode(message, env)
            self.receiver.receive_message(env)


In the same way we can define the receiving party.

.. code:: python
    class DenseReceiver:
        def __init__(self):
            self.buffer = EnvelopeBuffer()
            self.decoder = DenseDecoder()
            self.received_messages = []
    
        def receive_epr(self, env: Envelope):
            self.buffer.store(env)
    
        def receive_message(self, env: Envelope) -> tuple[int, int]:
            env_stored = self.buffer.get()
            ce = env.composite_envelope
            message = self.decoder.decode(env, env_stored)
            self.received_messages.append(message)


Lastly, we put the two parties to work, where we randomly generate the messages and in the end test whether the correct set of messages was received (it should be correct).

.. code:: python

    if __name__ == "__main__":
        NUMBER_OF_MESSAGES = 20
        sender = DenseSender()
        receiver = DenseReceiver()
        sender.register_receiver(receiver)
    
        # Generate message list
        messages = [ (randint(0, 1), randint(0, 1)) for _ in range(NUMBER_OF_MESSAGES)]
    
        # Preshare the entanglement
        sender.share_entanglement(NUMBER_OF_MESSAGES)
    
        # Send the messages super-densely
        for message in messages:
            sender.send_message(message)
    
        # Compare the received messages
        if messages == receiver.received_messages:
            print("Correctly encoded and decoded messages")
        else:
            print("Incorrectly encoded or decoded messages")
            print(messages)
            print(receiver.received_messages)

Running this protocol does indeed return the correct response message:

.. code:: bash

    $ python super_dense_coding.py                      
    Correctly encoded and decoded messages
