"""
Example of using Reusable components in order to
create super dense coding protocol
"""
from random import randint

from photon_weave.state.envelope import Envelope

from interoperable_devices import (
    DenseDecoder,
    DenseEncoder,
    EntangledPhotonSource,
    EnvelopeBuffer,
)


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


if __name__ == "__main__":
    NUMBER_OF_MESSAGES = 20
    sender = DenseSender()
    receiver = DenseReceiver()
    sender.register_receiver(receiver)

    # Generate message list
    messages = [(randint(0, 1), randint(0, 1)) for _ in range(NUMBER_OF_MESSAGES)]

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
