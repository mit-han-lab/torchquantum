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

__all__ = ["QuantumDevice"]


class QuantumDevice(nn.Module):
    def __init__(
        self,
        n_wires: int,
        device_name: str = "default",
        bsz: int = 1,
        device: Union[torch.device, str] = "cpu",
        record_op: bool = False,
    ):
        """A quantum device that contains the quantum state vector.
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

        _state = torch.zeros(2**self.n_wires, dtype=C_DTYPE)
        _state[0] = 1 + 0j  # type: ignore
        _state = torch.reshape(_state, [2] * self.n_wires).to(self.device)
        self.register_buffer("state", _state)

        repeat_times = [bsz] + [1] * len(self.state.shape)  # type: ignore
        self._states = self.state.repeat(*repeat_times)  # type: ignore
        self.register_buffer("states", self._states)

        self.record_op = record_op
        self.op_history = []

    def reset_op_history(self):
        """Resets the all Operation of the quantum device"""
        self.op_history = []

    def clone_states(self, existing_states: torch.Tensor):
        """Clone the states of the quantum device."""
        self.states = existing_states.clone()

    def set_states(self, states: torch.Tensor):
        """Set the states of the quantum device. The states are represented"""
        bsz = states.shape[0]
        self.states = torch.reshape(states, [bsz] + [2] * self.n_wires)

    def reset_states(self, bsz: int):
        """Reset the States of the quantum device"""
        repeat_times = [bsz] + [1] * len(self.state.shape)
        self.states = self.state.repeat(*repeat_times).to(self.state.device)

    def reset_identity_states(self):
        """Make the states as the identity matrix, one dim is the batch
        dim. Useful for verification.
        """
        self.states = torch.eye(
            2**self.n_wires, device=self.state.device, dtype=C_DTYPE
        ).reshape([2**self.n_wires] + [2] * self.n_wires)

    def reset_all_eq_states(self, bsz: int):
        """Make the states as the equal superposition state, one dim is the
        batch dim. Useful for verification.
        """
        energy = np.sqrt(1 / (2**self.n_wires) / 2)
        all_eq_state = torch.ones(2**self.n_wires, dtype=C_DTYPE) * (
            energy + energy * 1j
        )
        all_eq_state = all_eq_state.reshape([2] * self.n_wires)
        repeat_times = [bsz] + [1] * len(self.state.shape)
        self.states = all_eq_state.repeat(*repeat_times).to(self.state.device)

    def get_states_1d(self):
        """Return the states in a 1d tensor."""
        bsz = self.states.shape[0]
        return torch.reshape(self.states, [bsz, 2**self.n_wires])

    def get_state_1d(self):
        """Return the state in a 1d tensor."""
        return torch.reshape(self.state, [2**self.n_wires])

    @property
    def name(self):
        """Return the name of the device."""
        return self.__class__.__name__

    def __repr__(self):
        return f" class: {self.name} \n device name: {self.device_name} \n number of qubits: {self.n_wires} \n batch size: {self.bsz} \n current computing device: {self.state.device} \n recording op history: {self.record_op} \n current states: {repr(self.get_states_1d().cpu().detach().numpy())}"


for func_name, func in func_name_dict.items():
    setattr(QuantumDevice, func_name, func)

for func_name, func in func_name_dict_collect.items():
    setattr(QuantumDevice, func_name, func)
