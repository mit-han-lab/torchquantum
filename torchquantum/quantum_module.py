import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchquantum as tq

# class QuantumCircuit(object):
#     """Quantum circuit description
#
#     """

# input: phases, desception of circuit, (without x)
# output: unitary

# class builder(object)
#     def

# model parallel


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



class QuantumModule(nn.Module):
    r"""Quantum Module
    Arguments:

    Returns:
    """
    def __init__(self) -> None:
        super().__init__()

    # def forward(self, x_classical):
    #     self.q.RX(x[:10])
    #     q.rx(x[0])
    #
    #
    #     # statevect
    #     self.q.RXall(x)
    #
    #     builded_unitary = builder(phase*w, self.abc_descripton)
    #     #  statevect
    #     self.apply_unitary(builded_unitary)
    #
    #
    #     # get what is current state vector
    #     self.q.RX(x[1])


def test():
    pass


if __name__ == '__main__':
    test()
