import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

from typing import Iterable
from torchquantum.plugin.qiskit import QISKIT_INCOMPATIBLE_FUNC_NAMES
from torchpack.utils.logging import logger

class QFTLayer(tq.QuantumModule):
    def __init__(
        self,
        n_wires: int = None,
        wires: Iterable = None,
        do_swaps: bool = True,
        inverse: bool = False,
    ):
        """
        Constructs a Quantum Fourier Transform (QFT) layer

        Args:
            n_wires (int): Number of wires for the QFT as an integer
            wires (Iterable): Wires to perform the QFT as an Iterable
            do_swaps (bool): Whether or not to add the final swaps in a boolean format
            inverse (bool): Whether to create an inverse QFT layer in a boolean format
        """
        super().__init__()

        assert n_wires is not None or wires is not None

        if n_wires is None:
            self.n_wires = len(wires)

        if wires is None:
            wires = range(n_wires)

        self.n_wires = n_wires
        self.wires = wires
        self.do_swaps = do_swaps

        if inverse:
            self.gates_all = self.build_inverse_circuit()
        else:
            self.gates_all = self.build_circuit()

    def build_circuit(self):
        """Construct a QFT circuit."""

        operation_list = []

        # add the H and CU1 gates
        for top_wire in range(self.n_wires):
            operation_list.append({"name": "hadamard", "wires": self.wires[top_wire]})
            for wire in range(top_wire + 1, self.n_wires):
                lam = torch.pi / (2 ** (wire - top_wire))
                operation_list.append(
                    {
                        "name": "cu1",
                        "params": lam,
                        "wires": [self.wires[wire], self.wires[top_wire]],
                    }
                )

        # add swaps if specified
        if self.do_swaps:
            for wire in range(self.n_wires // 2):
                operation_list.append(
                    {
                        "name": "swap",
                        "wires": [
                            self.wires[wire],
                            self.wires[self.n_wires - wire - 1],
                        ],
                    }
                )

        return tq.QuantumModule.from_op_history(operation_list)

    def build_inverse_circuit(self):
        """Construct the inverse of a QFT circuit."""

        operation_list = []

        # add swaps if specified
        if self.do_swaps:
            for wire in range(self.n_wires // 2):
                operation_list.append(
                    {
                        "name": "swap",
                        "wires": [
                            self.wires[wire],
                            self.wires[self.n_wires - wire - 1],
                        ],
                    }
                )

        # add the CU1 and H gates
        for top_wire in range(self.n_wires)[::-1]:
            for wire in range(top_wire + 1, self.n_wires)[::-1]:
                lam = -torch.pi / (2 ** (wire - top_wire))
                operation_list.append(
                    {
                        "name": "cu1",
                        "params": lam,
                        "wires": [self.wires[wire], self.wires[top_wire]],
                    }
                )
            operation_list.append({"name": "hadamard", "wires": self.wires[top_wire]})

        return tq.QuantumModule.from_op_history(operation_list)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.gates_all(q_device)
