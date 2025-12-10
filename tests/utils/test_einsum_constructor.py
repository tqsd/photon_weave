import unittest

from photon_weave.state.fock import Fock
import photon_weave.extra.einsum_constructor as ESC


class TestTraceOut(unittest.TestCase):
    def test_trace_out_matrix(self) -> None:
        """
        Test the correctness of the trace out einsum string generation
        """
        fock1 = Fock()
        fock2 = Fock()
        fock3 = Fock()
        einsum = ESC.trace_out_matrix([fock1, fock2, fock3], [fock2])
        self.assertEqual(einsum, "abcadc->bd")
