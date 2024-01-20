"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

# from torchquantum.layer.layers import (
#     Op1QAllLayer,
#     RandomOp1All,
# )
from .op2_layer import Op2QAllLayer

from typing import Iterable
from torchquantum.plugin.qiskit import QISKIT_INCOMPATIBLE_FUNC_NAMES
from torchpack.utils.logging import logger

__all__ = [
    "EntangleLinear",
    "EntanglePairwise",
    "EntangleFull",
    "EntangleCircular",
    "Op2QDenseLayer",
    "EntanglementLayer",
]


class EntangleFull(tq.QuantumModule):
    """
    Quantum layer applying the same two-qubit operation in a dense pattern.

    This class represents a quantum layer that applies the same two-qubit operation in a dense pattern. The dense pattern connects every pair of wires, ensuring that each wire is connected to every other wire exactly once.

    Args:
        op (tq.Operator): Two-qubit operation to be applied.
        n_wires (int): Number of wires in the quantum device.
        has_params (bool, optional): Flag indicating if the operation has parameters. Defaults to False.
        trainable (bool, optional): Flag indicating if the operation is trainable. Defaults to False.
        wire_reverse (bool, optional): Flag indicating if the order of wires in each pair should be reversed. Defaults to False.

    """

    """pattern:
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]
    [1, 2], [1, 3], [1, 4], [1, 5]
    [2, 3], [2, 4], [2, 5]
    [3, 4], [3, 5]
    [4, 5]
    """

    def __init__(
        self, op, n_wires: int, has_params=False, trainable=False, wire_reverse=False
    ):
        super().__init__()
        self.n_wires = n_wires
        self.op = op
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        for k in range(self.n_wires * (self.n_wires - 1) // 2):
            self.ops_all.append(op(has_params=has_params, trainable=trainable))

    def forward(self, q_device):
        k = 0
        for i in range(self.n_wires - 1):
            for j in range(i + 1, self.n_wires):
                wires = [i, j]
                if self.wire_reverse:
                    wires.reverse()
                self.ops_all[k](q_device, wires=wires)
                k += 1


# Adding an alias to the previous name
Op2QDenseLayer = EntangleFull


class EntangleLinear(Op2QAllLayer):
    """
    Quantum layer applying the same two-qubit operation to all pairs of adjacent wires.
    This class represents a quantum layer that applies the same two-qubit operation to all pairs of adjacent wires
    in the quantum device.

    Args:
        op (tq.Operator): Two-qubit operation to be applied.
        n_wires (int): Number of wires in the quantum device.
        has_params (bool, optional): Flag indicating if the operation has parameters. Defaults to False.
        trainable (bool, optional): Flag indicating if the operation is trainable. Defaults to False.
        wire_reverse (bool, optional): Flag indicating if the order of wires in each pair should be reversed. Defaults to False.
    """

    """pattern: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]
    """

    def __init__(
        self,
        op,
        n_wires: int,
        has_params=False,
        trainable=False,
        wire_reverse=False,
    ):
        super().__init__(
            op=op,
            n_wires=n_wires,
            has_params=has_params,
            trainable=trainable,
            wire_reverse=wire_reverse,
            jump=1,
            circular=False,
        )


class EntangleCircular(Op2QAllLayer):
    """
    Quantum layer applying the same two-qubit operation to all pairs of adjacent wires in a circular manner.
    This class represents a quantum layer that applies the same two-qubit operation to all pairs of adjacent wires
    in the quantum device with a wrap-around

    Args:
        op (tq.Operator): Two-qubit operation to be applied.
        n_wires (int): Number of wires in the quantum device.
        has_params (bool, optional): Flag indicating if the operation has parameters. Defaults to False.
        trainable (bool, optional): Flag indicating if the operation is trainable. Defaults to False.
        wire_reverse (bool, optional): Flag indicating if the order of wires in each pair should be reversed. Defaults to False.
        jump (int, optional): Number of positions to jump between adjacent pairs of wires. Defaults to 1.
        circular (bool, optional): Flag indicating if the pattern should be circular. Defaults to False.

    """

    """pattern: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]
    """

    def __init__(
        self,
        op,
        n_wires: int,
        has_params=False,
        trainable=False,
        wire_reverse=False,
    ):
        super().__init__(
            op=op,
            n_wires=n_wires,
            has_params=has_params,
            trainable=trainable,
            wire_reverse=wire_reverse,
            jump=1,
            circular=True,
        )


class EntanglePairwise(tq.QuantumModule):
    """
    Quantum layer applying the same two-qubit operation in a pair-wise pattern

    This class represents a quantum layer that applies the same two-qubit operation in a pairwise pattern. The pairwise pattern first entangles all qubits i with i+1 for even i then all qubits i with i+1 for odd i.

    Args:
       op (tq.Operator): Two-qubit operation to be applied.
       n_wires (int): Number of wires in the quantum device.
       has_params (bool, optional): Flag indicating if the operation has parameters. Defaults to False.
       trainable (bool, optional): Flag indicating if the operation is trainable. Defaults to False.
       wire_reverse (bool, optional): Flag indicating if the order of wires in each pair should be reversed. Defaults to False.

    """

    """pattern:
    [0, 1], [2, 3], [4, 5]
    [1, 2], [3, 4] 
    """

    def __init__(
        self, op, n_wires: int, has_params=False, trainable=False, wire_reverse=False
    ):
        super().__init__()
        self.n_wires = n_wires
        self.op = op
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        for k in range(self.n_wires - 1):
            self.ops_all.append(op(has_params=has_params, trainable=trainable))

    def forward(self, q_device):
        k = 0

        # entangle qubit i with i+1 for all even values of i
        for i in range(self.n_wires - 1):
            if i % 2 == 0:
                wires = [i, i + 1]
                if self.wire_reverse:
                    wires.reverse()
                self.ops_all[k](q_device, wires=wires)
                k += 1

        # entangle qubit i with i+1 for all odd values of i
        for i in range(1, self.n_wires - 1):
            if i % 2 == 1:
                wires = [i, i + 1]
                if self.wire_reverse:
                    wires.reverse()
                self.ops_all[k](q_device, wires=wires)
                k += 1


class EntanglementLayer(tq.QuantumModule):
    """
    Quantum layer applying a specified two-qubit entanglement type to all qubits. The entanglement types include full, linear, pairwise, and circular.

    Args:
        op (tq.Operator): Two-qubit operation to be applied.
        n_wires (int): Number of wires in the quantum device.
        entanglement (str): Type of entanglement from ["full", "linear", "pairwise", "circular"]
        has_params (bool, optional): Flag indicating if the operation has parameters. Defaults to False.
        trainable (bool, optional): Flag indicating if the operation is trainable. Defaults to False.
        wire_reverse (bool, optional): Flag indicating if the order of wires in each pair should be reversed. Defaults to False.
        jump (int, optional): Number of positions to jump between adjacent pairs of wires. Defaults to 1.
        circular (bool, optional): Flag indicating if the pattern should be circular. Defaults to False.

    """

    def __init__(
        self,
        op,
        n_wires: int,
        entanglement: str,
        has_params=False,
        trainable=False,
        wire_reverse=False,
    ):
        super().__init__()

        entanglement_to_class = {
            "full": EntangleFull,
            "linear": EntangleLinear,
            "pairwise": EntanglePairwise,
            "circular": EntangleCircular,
        }

        self.entanglement_class = entanglement_to_class.get(entanglement, None)

        assert (
            self.entanglement_class is not None
        ), f"invalid entanglement type {entanglement}"

        self.entanglement_class.__init__(
            op=op,
            n_wires=n_wires,
            has_params=has_params,
            trainable=trainable,
            wire_reverse=wire_reverse,
        )

    @tq.static_support
    def forward(self, q_device):
        self.entanglement_class.forward(q_device)
