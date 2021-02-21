import torch
import torch.nn as nn

__all__ = ['QuantumDevice']


class QuantumDevice(nn.Module):
    def __init__(self, n_wire: int):
        super().__init__()
        # number of qubits
        self.n_wire = n_wire

        _state = torch.zeros(2 ** self.n_wire, dtype=torch.complex64)
        _state[0] = 1 + 0j
        _state = torch.reshape(_state, [2] * self.n_wire)
        self.register_buffer('state', _state)

        self.states = None

    def reset_states(self, bsz: int):
        repeat_times = [bsz] + [1] * len(self.state.shape)
        self.states = self.state.repeat(*repeat_times).to(
            self.state
        )

    def get_states_1d(self):
        bsz = self.states.shape[0]
        return torch.reshape(self.states, [bsz, 2 ** self.n_wire])

    def get_state_1d(self):
        return torch.reshape(self.state, [2 ** self.n_wire])

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.name} {self.n_wire} wires"
