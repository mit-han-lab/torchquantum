import torch
import torch.nn as nn
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
        self.states = self.state.repeat(*repeat_times).to(
            self.state.device
        )

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
