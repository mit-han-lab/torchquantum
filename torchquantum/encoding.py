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


class GeneralEncoder(Encoder, metaclass=ABCMeta):
    """func_list list of dict
    Example 1:
    [
    {'input_idx': [0], 'func': 'rx', 'wires': [0]},
    {'input_idx': [1], 'func': 'rx', 'wires': [1]},
    {'input_idx': [2], 'func': 'rx', 'wires': [2]},
    {'input_idx': [3], 'func': 'rx', 'wires': [3]},
    {'input_idx': [4], 'func': 'ry', 'wires': [0]},
    {'input_idx': [5], 'func': 'ry', 'wires': [1]},
    {'input_idx': [6], 'func': 'ry', 'wires': [2]},
    {'input_idx': [7], 'func': 'ry', 'wires': [3]},
    {'input_idx': [8], 'func': 'rz', 'wires': [0]},
    {'input_idx': [9], 'func': 'rz', 'wires': [1]},
    {'input_idx': [10], 'func': 'rz', 'wires': [2]},
    {'input_idx': [11], 'func': 'rz', 'wires': [3]},
    {'input_idx': [12], 'func': 'rx', 'wires': [0]},
    {'input_idx': [13], 'func': 'rx', 'wires': [1]},
    {'input_idx': [14], 'func': 'rx', 'wires': [2]},
    {'input_idx': [15], 'func': 'rx', 'wires': [3]},
    ]

    Example 2:
    [
    {'input_idx': [0, 1, 2], 'func': 'u3', 'wires': [0]},
    {'input_idx': [3], 'func': 'u1', 'wires': [0]},
    {'input_idx': [4, 5, 6], 'func': 'u3', 'wires': [1]},
    {'input_idx': [7], 'func': 'u1', 'wires': [1]},
    {'input_idx': [8, 9, 10], 'func': 'u3', 'wires': [2]},
    {'input_idx': [11], 'func': 'u1', 'wires': [2]},
    {'input_idx': [12, 13, 14], 'func': 'u3', 'wires': [3]},
    {'input_idx': [15], 'func': 'u1', 'wires': [3]},
    ]
    """
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        self.q_device.reset_states(x.shape[0])
        for info in self.func_list:
            func_name_dict[info['func']](
                self.q_device,
                wires=info['wires'],
                params=x[:, info['input_idx']],
                static=self.static_mode,
                parent_graph=self.graph
            )


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


encoder_op_list_name_dict = {
    '4x4_ryzxy':
        [
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
            {'input_idx': [2], 'func': 'ry', 'wires': [2]},
            {'input_idx': [3], 'func': 'ry', 'wires': [3]},
            {'input_idx': [4], 'func': 'rz', 'wires': [0]},
            {'input_idx': [5], 'func': 'rz', 'wires': [1]},
            {'input_idx': [6], 'func': 'rz', 'wires': [2]},
            {'input_idx': [7], 'func': 'rz', 'wires': [3]},
            {'input_idx': [8], 'func': 'rx', 'wires': [0]},
            {'input_idx': [9], 'func': 'rx', 'wires': [1]},
            {'input_idx': [10], 'func': 'rx', 'wires': [2]},
            {'input_idx': [11], 'func': 'rx', 'wires': [3]},
            {'input_idx': [12], 'func': 'ry', 'wires': [0]},
            {'input_idx': [13], 'func': 'ry', 'wires': [1]},
            {'input_idx': [14], 'func': 'ry', 'wires': [2]},
            {'input_idx': [15], 'func': 'ry', 'wires': [3]}
        ],
    '6x6_ryzxy':
        [
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
            {'input_idx': [2], 'func': 'ry', 'wires': [2]},
            {'input_idx': [3], 'func': 'ry', 'wires': [3]},
            {'input_idx': [4], 'func': 'ry', 'wires': [4]},
            {'input_idx': [5], 'func': 'ry', 'wires': [5]},
            {'input_idx': [6], 'func': 'ry', 'wires': [6]},
            {'input_idx': [7], 'func': 'ry', 'wires': [7]},
            {'input_idx': [8], 'func': 'ry', 'wires': [8]},
            {'input_idx': [9], 'func': 'ry', 'wires': [9]},
            {'input_idx': [10], 'func': 'rz', 'wires': [0]},
            {'input_idx': [11], 'func': 'rz', 'wires': [1]},
            {'input_idx': [12], 'func': 'rz', 'wires': [2]},
            {'input_idx': [13], 'func': 'rz', 'wires': [3]},
            {'input_idx': [14], 'func': 'rz', 'wires': [4]},
            {'input_idx': [15], 'func': 'rz', 'wires': [5]},
            {'input_idx': [16], 'func': 'rz', 'wires': [6]},
            {'input_idx': [17], 'func': 'rz', 'wires': [7]},
            {'input_idx': [18], 'func': 'rz', 'wires': [8]},
            {'input_idx': [19], 'func': 'rz', 'wires': [9]},
            {'input_idx': [20], 'func': 'rx', 'wires': [0]},
            {'input_idx': [21], 'func': 'rx', 'wires': [1]},
            {'input_idx': [22], 'func': 'rx', 'wires': [2]},
            {'input_idx': [23], 'func': 'rx', 'wires': [3]},
            {'input_idx': [24], 'func': 'rx', 'wires': [4]},
            {'input_idx': [25], 'func': 'rx', 'wires': [5]},
            {'input_idx': [26], 'func': 'rx', 'wires': [6]},
            {'input_idx': [27], 'func': 'rx', 'wires': [7]},
            {'input_idx': [28], 'func': 'rx', 'wires': [8]},
            {'input_idx': [29], 'func': 'rx', 'wires': [9]},
            {'input_idx': [30], 'func': 'ry', 'wires': [0]},
            {'input_idx': [31], 'func': 'ry', 'wires': [1]},
            {'input_idx': [32], 'func': 'ry', 'wires': [2]},
            {'input_idx': [33], 'func': 'ry', 'wires': [3]},
            {'input_idx': [34], 'func': 'ry', 'wires': [4]},
            {'input_idx': [35], 'func': 'ry', 'wires': [5]},
        ]
}
