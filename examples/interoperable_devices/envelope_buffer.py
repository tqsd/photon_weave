"""
Envelope buffer stores preshared entangled photons
"""
import queue
from photon_weave.state.envelope import Envelope


class EnvelopeBuffer:
    def __init__(self):
        self.buffer = queue.Queue()

    def store(self, env: Envelope):
        self.buffer.put(env)

    def get(self) -> Envelope:
        return self.buffer.get()
