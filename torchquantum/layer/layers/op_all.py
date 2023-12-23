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
    "TrainableOpAll",
    "ClassicalInOpAll",
    "FixedOpAll",
    "TwoQAll",
]


class TrainableOpAll(tq.QuantumModule):
    """Rotation rx on all qubits
    The rotation angle is a parameter of each rotation gate
    One potential optimization is to compute the unitary of all gates
    together.
    """

    def __init__(self, n_gate: int, op: tq.Operation):
        super().__init__()
        self.n_gate = n_gate
        self.gate_all = nn.ModuleList()
        for k in range(self.n_gate):
            self.gate_all.append(op(has_params=True, trainable=True))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        # rx on all wires, assert the number of gate is the same as the number
        # of wires in the device.
        assert self.n_gate == q_device.n_wires, (
            f"Number of rx gates ({self.n_gate}) is different from number "
            f"of wires ({q_device.n_wires})!"
        )

        for k in range(self.n_gate):
            self.gate_all[k](q_device, wires=k)


class ClassicalInOpAll(tq.QuantumModule):
    """
    Quantum module that applies the same quantum operation to all wires of a quantum device,
    where the parameters of the operation are obtained from a classical input.

    Args:
        n_gate (int): Number of gates.
        op (tq.Operator): Quantum operation to be applied.

    Attributes:
        n_gate (int): Number of gates.
        gate_all (nn.ModuleList): List of quantum operations.

    """

    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.n_gate = n_gate
        self.gate_all = nn.ModuleList()
        for k in range(self.n_gate):
            self.gate_all.append(op())

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        """
        Performs the forward pass of the classical input quantum operation module.

        Args:
            q_device (tq.QuantumDevice): Quantum device to apply the operations on.
            x (torch.Tensor): Classical input of shape (batch_size, n_gate).

        Returns:
            None

        Raises:
            AssertionError: If the number of gates is different from the number of wires in the device.

        """
        # rx on all wires, assert the number of gate is the same as the number
        # of wires in the device.
        assert self.n_gate == q_device.n_wires, (
            f"Number of rx gates ({self.n_gate}) is different from number "
            f"of wires ({q_device.n_wires})!"
        )

        for k in range(self.n_gate):
            self.gate_all[k](q_device, wires=k, params=x[:, k])


class FixedOpAll(tq.QuantumModule):
    """
    Quantum module that applies the same fixed quantum operation to all wires of a quantum device.

    Args:
        n_gate (int): Number of gates.
        op (tq.Operator): Quantum operation to be applied.

    Attributes:
        n_gate (int): Number of gates.
        gate_all (nn.ModuleList): List of quantum operations.

    """

    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.n_gate = n_gate
        self.gate_all = nn.ModuleList()
        for k in range(self.n_gate):
            self.gate_all.append(op())

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        """
        Performs the forward pass of the fixed quantum operation module.

        Args:
            q_device (tq.QuantumDevice): Quantum device to apply the operations on.

        Returns:
            None

        Raises:
            AssertionError: If the number of gates is different from the number of wires in the device.

        """
        # rx on all wires, assert the number of gate is the same as the number
        # of wires in the device.
        assert self.n_gate == q_device.n_wires, (
            f"Number of rx gates ({self.n_gate}) is different from number "
            f"of wires ({q_device.n_wires})!"
        )

        for k in range(self.n_gate):
            self.gate_all[k](q_device, wires=k)


class TwoQAll(tq.QuantumModule):
    """
    Quantum module that applies a two-qubit quantum operation to adjacent pairs of wires in a quantum device.

    Args:
        n_gate (int): Number of adjacent pairs of wires.
        op (tq.Operator): Two-qubit quantum operation to be applied.

    Attributes:
        n_gate (int): Number of adjacent pairs of wires.
        op (tq.Operator): Two-qubit quantum operation.

    """

    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.n_gate = n_gate
        self.op = op()

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        for k in range(self.n_gate - 1):
            self.op(q_device, wires=[k, k + 1])
        self.op(q_device, wires=[self.n_gate - 1, 0])
