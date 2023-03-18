# import torchquantum as tq
# import torchquantum.functional as tqf
# import torch
# import torch.nn.functional as F
# import numpy as np
# import os
# import random
# import datetime
#
# from torchquantum.encoding import encoder_op_list_name_dict
# from torchpack.utils.logging import logger
# from torchquantum.layers import layer_name_dict
# from torchpack.utils.config import configs
# import pdb
#
#
# class QVQEModel0(tq.QuantumModule):
#     def __init__(self, arch, hamil_info):
#         super().__init__()
#         self.arch = arch
#         self.hamil_info = hamil_info
#         self.n_wires = arch['n_wires']
#         self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
#         self.q_layer = layer_name_dict[arch['q_layer_name']](arch)
#         self.measure = tq.MeasureMultipleTimes(
#             obs_list=hamil_info['hamil_list'])
#         self.num_forwards = 0
#         self.n_params = len(list(self.parameters()))
#
#     def forward(self, x, verbose=False, use_qiskit=False):
#         if use_qiskit:
#             x = self.qiskit_processor.process_multi_measure(
#                 self.q_device, self.q_layer, self.measure)
#         else:
#             self.q_device.reset_states(bsz=1)
#             self.q_layer(self.q_device)
#             x = self.measure(self.q_device)
#
#         hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
#                                            self.hamil_info['hamil_list']],
#                                           device=x.device).double()
#
#         for k, hamil in enumerate(self.hamil_info['hamil_list']):
#             for wire, observable in zip(hamil['wires'], hamil['observables']):
#                 if observable == 'i':
#                     x[k][wire] = 1
#             for wire in range(self.q_device.n_wires):
#                 if wire not in hamil['wires']:
#                     x[k][wire] = 1
#
#         if verbose:
#             logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")
#
#         x = torch.cumprod(x, dim=-1)[:, -1].double()
#         x = torch.dot(x, hamil_coefficients)
#
#         if x.dim() == 0:
#             x = x.unsqueeze(0)
#
#         return x
#
#     def shift_and_run(self, x, global_step, total_step, verbose=False, use_qiskit=False):
#         with torch.no_grad():
#             if use_qiskit:
#                 self.q_device.reset_states(bsz=1)
#                 x = self.qiskit_processor.process_multi_measure(
#                     self.q_device, self.q_layer, self.measure)
#                 self.grad_list = []
#                 for i, param in enumerate(self.parameters()):
#                     param.copy_(param + np.pi*0.5)
#                     out1 = self.qiskit_processor.process_multi_measure(
#                         self.q_device, self.q_layer, self.measure)
#                     param.copy_(param - np.pi)
#                     out2 = self.qiskit_processor.process_multi_measure(
#                         self.q_device, self.q_layer, self.measure)
#                     param.copy_(param + np.pi*0.5)
#                     grad = 0.5 * (out1 - out2)
#                     self.grad_list.append(grad)
#             else:
#                 self.q_device.reset_states(bsz=1)
#                 self.q_layer(self.q_device)
#                 x = self.measure(self.q_device)
#                 self.grad_list = []
#                 for i, param in enumerate(self.parameters()):
#                     param.copy_(param + np.pi*0.5)
#                     self.q_device.reset_states(bsz=1)
#                     self.q_layer(self.q_device)
#                     out1 = self.measure(self.q_device)
#                     param.copy_(param - np.pi)
#                     self.q_device.reset_states(bsz=1)
#                     self.q_layer(self.q_device)
#                     out2 = self.measure(self.q_device)
#                     param.copy_(param + np.pi*0.5)
#                     grad = 0.5 * (out1 - out2)
#                     self.grad_list.append(grad)
#
#         hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
#                                            self.hamil_info['hamil_list']],
#                                           device=x.device).double()
#         x_axis = []
#         y_axis = []
#         for k, hamil in enumerate(self.hamil_info['hamil_list']):
#             for wire, observable in zip(hamil['wires'], hamil['observables']):
#                 if observable == 'i':
#                     x[k][wire] = 1
#                     x_axis.append(k)
#                     y_axis.append(wire)
#             for wire in range(self.q_device.n_wires):
#                 if wire not in hamil['wires']:
#                     x[k][wire] = 1
#                     x_axis.append(k)
#                     y_axis.append(wire)
#         for grad in self.grad_list:
#             grad[x_axis, y_axis] = 0
#
#         self.circuit_output = x
#         self.circuit_output.requires_grad = True
#         self.num_forwards += (1 + 2 * self.n_params) * x.shape[0]
#         if verbose:
#             logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")
#
#         x = torch.cumprod(x, dim=-1)[:, -1].double()
#         x = torch.dot(x, hamil_coefficients)
#
#         if x.dim() == 0:
#             x = x.unsqueeze(0)
#
#         return x
#
#     def backprop_grad(self):
#         for i, param in enumerate(self.parameters()):
#             param.grad = torch.sum(self.grad_list[i] * self.circuit_output.grad).to(dtype=torch.float32).view(param.shape)
#
# model_dict = {
#     'vqe_0': QVQEModel0,
# }

import torch
import torch.nn as nn
import numpy as np

from typing import Union, List, Iterable


__all__ = [
    "QuantumPulse",
    "QuantumPulseGaussian",
    "QuantumPulseDirect",
]


class QuantumPulse(nn.Module):
    """
    The Quantum Pulse simulator
    """

    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


class QuantumPulseDirect(QuantumPulse):
    def __init__(
        self,
        n_steps: int,
        hamil,
        delta_t: float = 1.0,
        initial_shape: List[float] = None,
    ):
        super().__init__()
        self.hamil = torch.tensor(hamil, dtype=torch.complex64)
        if initial_shape is not None:
            assert len(initial_shape) == n_steps
            initial_shape = torch.Tensor(initial_shape)
        else:
            initial_shape = torch.ones(n_steps)
        self.pulse_shape = nn.Parameter(initial_shape)
        self.n_steps = n_steps
        self.delta_t = delta_t

    def get_unitary(self):
        unitary_per_step = []
        for k in range(self.n_steps):
            magnitude = self.pulse_shape[k]
            unitary = torch.matrix_exp(-1j * self.hamil * magnitude * self.delta_t)
            # print(unitary @ unitary.conj().T)
            # unitary_mag = (unitary[0]**2).sum().sqrt()
            # unitary = unitary_mag / unitary

            unitary_per_step.append(unitary)

        u_overall = None
        for k, u in enumerate(unitary_per_step):
            if not k:
                u_overall = u
            else:
                u_overall = u_overall @ u

        return u_overall

    def __repr__(self):
        return f"QuantumPulse Direct \n shape: {self.pulse_shape}"


class QuantumPulseGaussian(QuantumPulse):
    """Gaussian Quantum Pulse, will only count +- five sigmas"""

    def __init__(
        self,
        hamil,
        n_steps: int = 100,
        delta_t: float = 1.0,
        x_min: float = -10,
        x_max: float = 10,
        initial_params: List[float] = None,
    ):
        super(QuantumPulseGaussian, self).__init__()
        self.hamil = torch.tensor(hamil, dtype=torch.complex64)
        self.delta_t = delta_t
        # mag, mu, sigma
        if initial_params is not None:
            assert len(initial_params) == 3
            initial_params = torch.Tensor(initial_params)
        else:
            initial_params = torch.ones(3)

        self.pulse_params = nn.Parameter(initial_params)
        self.n_steps = n_steps
        self.delta_x = (x_max - x_min) / n_steps
        self.x_list = torch.tensor(np.arange(x_min, x_max, self.delta_x))

    def get_unitary(self):
        self.mag = self.pulse_params[0]
        self.mu = self.pulse_params[1]
        self.sigma = self.pulse_params[2]

        # delta_x = (10 * self.sigma / self.n_steps).item()
        # self.x_list = torch.tensor(np.arange(
        # (self.mu - 5 * self.sigma).item(),
        # (self.mu + 5 * self.sigma).item(), delta_x))

        self.pulse_shape = self.mag * torch.exp(
            -((self.x_list - self.mu) ** 2) / (2 * self.sigma**2)
        )

        unitary_per_step = []
        for k in range(self.n_steps):
            magnitude = self.pulse_shape[k]
            unitary = torch.matrix_exp(
                -1j * self.hamil * magnitude * self.delta_t * self.delta_x
            )
            # print(unitary @ unitary.conj().T)
            # unitary_mag = (unitary[0]**2).sum().sqrt()
            # unitary = unitary_mag / unitary

            unitary_per_step.append(unitary)

        u_overall = None
        for k, u in enumerate(unitary_per_step):
            if not k:
                u_overall = u
            else:
                u_overall = u_overall @ u

        return u_overall

    def __repr__(self):
        return f"QuantumPulse Guassian \n shape: {self.pulse_shape}"


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    pulse = QuantumPulseDirect(n_steps=10, hamil=[[0, 1], [1, 0]])

    print(pulse.get_unitary())

    print("finish")
