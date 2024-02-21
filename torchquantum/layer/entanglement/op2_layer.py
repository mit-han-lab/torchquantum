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


from typing import Iterable
from torchquantum.plugin.qiskit import QISKIT_INCOMPATIBLE_FUNC_NAMES
from torchpack.utils.logging import logger

__all__ = [
    "Op2QAllLayer",
    "Op2QFit32Layer",
    "Op2QButterflyLayer",
]


class Op2QAllLayer(tq.QuantumModule):
    """
    Quantum layer applying the same two-qubit operation to all pairs of adjacent wires.
    This class represents a quantum layer that applies the same two-qubit operation to all pairs of adjacent wires
    in the quantum device. The pairs of wires can be determined in a circular or non-circular pattern based on the
    specified jump.

    Args:
        op (tq.Operator): Two-qubit operation to be applied.
        n_wires (int): Number of wires in the quantum device.
        has_params (bool, optional): Flag indicating if the operation has parameters. Defaults to False.
        trainable (bool, optional): Flag indicating if the operation is trainable. Defaults to False.
        wire_reverse (bool, optional): Flag indicating if the order of wires in each pair should be reversed. Defaults to False.
        jump (int, optional): Number of positions to jump between adjacent pairs of wires. Defaults to 1.
        circular (bool, optional): Flag indicating if the pattern should be circular. Defaults to False.

    """

    """pattern:
    circular = False
    jump = 1: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]
    jump = 2: [0, 2], [1, 3], [2, 4], [3, 5]
    jump = 3: [0, 3], [1, 4], [2, 5]
    jump = 4: [0, 4], [1, 5]
    jump = 5: [0, 5]

    circular = True
    jump = 1: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]
    jump = 2: [0, 2], [1, 3], [2, 4], [3, 5], [4, 0], [5, 1]
    jump = 3: [0, 3], [1, 4], [2, 5], [3, 0], [4, 1], [5, 2]
    jump = 4: [0, 4], [1, 5], [2, 0], [3, 1], [4, 2], [5, 3]
    jump = 5: [0, 5], [1, 0], [2, 1], [3, 2], [4, 3], [5, 4]
    """

    def __init__(
        self,
        op,
        n_wires: int,
        has_params=False,
        trainable=False,
        wire_reverse=False,
        jump=1,
        circular=False,
    ):
        super().__init__()
        self.n_wires = n_wires
        self.jump = jump
        self.circular = circular
        self.op = op
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        if circular:
            n_ops = n_wires
        else:
            n_ops = n_wires - jump
        for k in range(n_ops):
            self.ops_all.append(op(has_params=has_params, trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(len(self.ops_all)):
            wires = [k, (k + self.jump) % self.n_wires]
            if self.wire_reverse:
                wires.reverse()
            self.ops_all[k](q_device, wires=wires)


class Op2QFit32Layer(tq.QuantumModule):
    """
    Quantum layer applying the same two-qubit operation to all pairs of adjacent wires, fitting to 32 operations.

    This class represents a quantum layer that applies the same two-qubit operation to all pairs of adjacent wires in the quantum device. The pairs of wires can be determined in a circular or non-circular pattern based on the specified jump. The layer is designed to fit to 32 operations by repeating the same operation pattern multiple times.

    Args:
        op (tq.Operator): Two-qubit operation to be applied.
        n_wires (int): Number of wires in the quantum device.
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
        has_params=False,
        trainable=False,
        wire_reverse=False,
        jump=1,
        circular=False,
    ):
        super().__init__()
        self.n_wires = n_wires
        self.jump = jump
        self.circular = circular
        self.op = op
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        # if circular:
        #     n_ops = n_wires
        # else:
        #     n_ops = n_wires - jump
        n_ops = 32
        for k in range(n_ops):
            self.ops_all.append(op(has_params=has_params, trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(len(self.ops_all)):
            wires = [k % self.n_wires, (k + self.jump) % self.n_wires]
            if self.wire_reverse:
                wires.reverse()
            self.ops_all[k](q_device, wires=wires)


class Op2QButterflyLayer(tq.QuantumModule):
    """
    Quantum layer applying the same two-qubit operation in a butterfly pattern.

    This class represents a quantum layer that applies the same two-qubit operation in a butterfly pattern. The butterfly pattern connects the first and last wire, the second and second-to-last wire, and so on, until the center wire(s) in the case of an odd number of wires.

    Args:
        op (tq.Operator): Two-qubit operation to be applied.
        n_wires (int): Number of wires in the quantum device.
        has_params (bool, optional): Flag indicating if the operation has parameters. Defaults to False.
        trainable (bool, optional): Flag indicating if the operation is trainable. Defaults to False.
        wire_reverse (bool, optional): Flag indicating if the order of wires in each pair should be reversed. Defaults to False.

    """

    """pattern: [0, 5], [1, 4], [2, 3]"""

    def __init__(
        self, op, n_wires: int, has_params=False, trainable=False, wire_reverse=False
    ):
        super().__init__()
        self.n_wires = n_wires
        self.op = op
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        for k in range(n_wires // 2):
            self.ops_all.append(op(has_params=has_params, trainable=trainable))

    def forward(self, q_device):
        for k in range(len(self.ops_all)):
            wires = [k, self.n_wires - 1 - k]
            if self.wire_reverse:
                wires.reverse()
            self.ops_all[k](q_device, wires=wires)
