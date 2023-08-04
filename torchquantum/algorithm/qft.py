import torchquantum as tq
from typing import Iterable

__all__ = ["QFT"]


class QFT(object):
    def __init__(
        self, n_wires: int = None, wires: Iterable = None, do_swaps=True
    ) -> None:
        """Init function for QFT class

        Args:
            n_wires (int): Number of wires for the QFT as an integer
            wires (Iterable): Wires to perform the QFT as an Iterable
            add_swaps (bool): Whether or not to add the final swaps in a boolean format
            inverse (bool): Whether to create an inverse QFT layer in a boolean format
        """
        super().__init__()

        self.n_wires = n_wires
        self.wires = wires
        self.do_swaps = do_swaps

    def construct_qft_circuit(self) -> tq.QuantumModule:
        """Construct the QFT circuit."""
        return tq.layer.QFTLayer(
            n_wires=self.n_wires, wires=self.wires, do_swaps=self.do_swaps
        )

    def construct_inverse_qft_circuit(self) -> tq.QuantumModule:
        """Construct the inverse of a QFT circuit."""
        return tq.layer.QFTLayer(
            n_wires=self.n_wires, wires=self.wires, do_swaps=self.do_swaps, inverse=True
        )
