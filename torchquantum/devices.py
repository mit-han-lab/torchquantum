import torch
import torch.nn as nn
import numpy as np

from torchquantum.macro import C_DTYPE

__all__ = ['QuantumDevice']


class QuantumDevice(nn.Module):
    def __init__(self, n_wires: int):
        super().__init__()
        # number of qubits
        self.n_wires = n_wires

        _state = torch.zeros(2 ** self.n_wires, dtype=C_DTYPE)
        _state[0] = 1 + 0j
        _state = torch.reshape(_state, [2] * self.n_wires)
        self.register_buffer('state', _state)

        self.states = None

    def reset_states(self, bsz: int):
        repeat_times = [bsz] + [1] * len(self.state.shape)
        self.states = self.state.repeat(*repeat_times).to(self.state.device)

    def reset_identity_states(self):
        """Make the states as the identity matrix, one dim is the batch
        dim. Useful for verification.
        """
        self.states = torch.eye(2 ** self.n_wires, device=self.state.device,
                                dtype=C_DTYPE).reshape([2 ** self.n_wires] +
                                                       [2] * self.n_wires)

    def reset_all_eq_states(self, bsz: int):
        energy = np.sqrt(1 / (2 ** self.n_wires) / 2)
        all_eq_state = torch.ones(2 ** self.n_wires, dtype=C_DTYPE) * \
            (energy + energy * 1j)
        all_eq_state = all_eq_state.reshape([2] * self.n_wires)
        repeat_times = [bsz] + [1] * len(self.state.shape)
        self.states = all_eq_state.repeat(*repeat_times).to(self.state.device)

    def get_states_1d(self):
        bsz = self.states.shape[0]
        return torch.reshape(self.states, [bsz, 2 ** self.n_wires])

    def get_state_1d(self):
        return torch.reshape(self.state, [2 ** self.n_wires])

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.name} {self.n_wires} wires"
