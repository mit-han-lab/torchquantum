import torch
import torchquantum as tq

from typing import Iterable
from torchquantum.macro import C_DTYPE
from abc import ABCMeta


class Encoder(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, q_device: tq.QuantumDevice, x):
        raise NotImplementedError


class PhaseEncoder(Encoder, metaclass=ABCMeta):
    def __init__(self, func):
        super().__init__()
        self.func = func

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        self.q_device.reset_states(x.shape[0])
        for k in range(self.q_device.n_wires):
            self.func(self.q_device, wires=k, params=x[:, k],
                      static=self.static_mode, parent_graph=self.graph)


class MultiPhaseEncoder(Encoder, metaclass=ABCMeta):
    def __init__(self, func):
        super().__init__()
        self.func = func if isinstance(func, Iterable) else [func]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        self.q_device.reset_states(x.shape[0])
        for k, func in enumerate(self.func):
            func(self.q_device, wires=k % self.q_device.n_wires,
                 params=x[:, k], static=self.static_mode,
                 parent_graph=self.graph)


class StateEncoder(Encoder, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def forward(self, q_device: tq.QuantumDevice, x):
        # encoder the x to the statevector of the quantum device
        self.q_device = q_device

        # normalize the input
        x = x / (torch.sqrt((x * x).sum(dim=-1))).unsqueeze(-1)
        state = torch.cat((x, torch.zeros(
            x.shape[0], 2 ** self.q_device.n_wires - x.shape[1],
            device=x.device)), dim=-1)
        state = state.view([x.shape[0]] + [2] * self.q_device.n_wires)

        self.q_device.states = state.type(C_DTYPE)


class MagnitudeEncoder(Encoder, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
