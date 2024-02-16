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
import numpy as np

from torchquantum.macro import C_DTYPE
from torchquantum.functional import func_name_dict, func_name_dict_collect
from typing import Union

__all__ = ["NoiseDevice"]


class NoiseDevice(nn.Module):
    def __init__(
            self,
            n_wires: int,
            device_name: str = "noisedevice",
            bsz: int = 1,
            device: Union[torch.device, str] = "cpu",
            record_op: bool = False,
    ):
        """A quantum device that support the density matrix simulation
        Args:
            n_wires: number of qubits
            device_name: name of the quantum device
            bsz: batch size of the quantum state
            device: which classical computing device to use, 'cpu' or 'cuda'
            record_op: whether to record the operations on the quantum device and then
                they can be used to construct a static computation graph
        """
        super().__init__()
        # number of qubits
        # the states are represented in a multi-dimension tensor
        # from left to right: qubit 0 to n
        self.n_wires = n_wires
        self.device_name = device_name
        self.bsz = bsz
        self.device = device

        _density = torch.zeros(2 ** (2 * self.n_wires), dtype=C_DTYPE)
        _density[0] = 1 + 0j
        _density = torch.reshape(_density, [2] * (2 * self.n_wires))
        self._dims = 2 * self.n_wires
        self.register_buffer("density", _density)

        repeat_times = [bsz] + [1] * len(self.density.shape)  # type: ignore
        self._densities = self.density.repeat(*repeat_times)  # type: ignore
        self.register_buffer("densities", self._densities)

        self.record_op = record_op
        self.op_history = []

    def reset_op_history(self):
        """Resets the all Operation of the quantum device"""
        self.op_history = []

    def print_2d(self, index):
        """Print the matrix value at the given index.

        This method prints the matrix value of `matrix[index]`. It reshapes the value into a 2D matrix
        using the `torch.reshape` function and then prints it.

        Args:
            index (int): The index of the matrix value to print.

        Examples:
            >>> device = QuantumDevice(n_wires=2)
            >>> device.matrix = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            >>> device.print_2d(1)
            tensor([[0, 0],
                    [0, 1]])

        """

        _matrix = torch.reshape(self.densities[index], [2 ** self.n_wires] * 2)
        print(_matrix)

    def get_2d_matrix(self, index):
        _matrix = torch.reshape(self.densities[index], [2 ** self.n_wires] * 2)
        return _matrix

    def calc_trace(self, index):
        _matrix = torch.reshape(self.densities[index], [2 ** self.n_wires] * 2)
        return torch.trace(_matrix)

    @property
    def name(self):
        """Return the name of the device."""
        return self.__class__.__name__

    def __repr__(self):
        return f" class: {self.name} \n device name: {self.device_name} \n number of qubits: {self.n_wires} \n batch size: {self.bsz} \n current computing device: {self.density.device} \n recording op history: {self.record_op} \n current states: {repr(self.get_probs_1d().cpu().detach().numpy())}"

    '''
    Get the probability of measuring each state to a one dimension
    tensor
    '''

    def get_probs_1d(self):
        """Return the states in a 1d tensor."""
        bsz = self.densities.shape[0]
        densities2d = torch.reshape(self.densities, [bsz, 2 ** self.n_wires, 2 ** self.n_wires])
        return torch.abs(torch.diagonal(densities2d, offset=0, dim1=1, dim2=2))

    def get_prob_1d(self):
        """Return the state in a 1d tensor."""
        density2d = torch.reshape(self.density, [2 ** self.n_wires, 2 ** self.n_wires])
        return torch.abs(torch.diagonal(density2d, offset=0, dim1=0, dim2=1))

    def clone_densities(self, existing_densities: torch.Tensor):
        """Clone the densities of the other quantum device."""
        self.densities = existing_densities.clone()


for func_name, func in func_name_dict.items():
    setattr(NoiseDevice, func_name, func)

for func_name, func in func_name_dict_collect.items():
    setattr(NoiseDevice, func_name, func)
