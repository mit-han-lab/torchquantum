import torch
import torchquantum as tq

from torchquantum.functional import func_name_dict
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
    def __init__(self, funcs, wires=None):
        super().__init__()
        self.funcs = funcs if isinstance(funcs, Iterable) else [funcs]
        self.wires = wires

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        if self.wires is None:
            self.wires = list(range(q_device.n_wires)) * (len(self.funcs) //
                                                          q_device.n_wires)
        self.q_device = q_device
        self.q_device.reset_states(x.shape[0])

        x_id = 0
        for k, func in enumerate(self.funcs):
            if func in ['rx', 'ry', 'rz', 'u1', 'phaseshift']:
                stride = 1
            elif func == 'u2':
                stride = 2
            elif func == 'u3':
                stride = 3
            else:
                raise ValueError(func)

            func_name_dict[func](self.q_device, wires=self.wires[k],
                                 params=x[:, x_id:(x_id + stride)],
                                 static=self.static_mode,
                                 parent_graph=self.graph)
            x_id += stride


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
