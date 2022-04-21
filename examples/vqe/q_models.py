import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import datetime

from torchquantum.encoding import encoder_op_list_name_dict
from torchpack.utils.logging import logger
from torchquantum.layers import layer_name_dict
from torchpack.utils.config import configs
import pdb


class QVQEModel0(tq.QuantumModule):
    def __init__(self, arch, hamil_info):
        super().__init__()
        self.arch = arch
        self.hamil_info = hamil_info
        self.n_wires = arch['n_wires']
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = layer_name_dict[arch['q_layer_name']](arch)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])
        self.num_forwards = 0
        self.n_params = len(list(self.parameters()))

    def forward(self, x, verbose=False, use_qiskit=False):
        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                           self.hamil_info['hamil_list']],
                                          device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def shift_and_run(self, x, global_step, total_step, verbose=False, use_qiskit=False):
        with torch.no_grad():
            if use_qiskit:
                self.q_device.reset_states(bsz=1)
                x = self.qiskit_processor.process_multi_measure(
                    self.q_device, self.q_layer, self.measure)
                self.grad_list = []
                for i, param in enumerate(self.parameters()):
                    param.copy_(param + np.pi*0.5)
                    out1 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param - np.pi)
                    out2 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)
            else:
                self.q_device.reset_states(bsz=1)
                self.q_layer(self.q_device)
                x = self.measure(self.q_device)
                self.grad_list = []
                for i, param in enumerate(self.parameters()):
                    param.copy_(param + np.pi*0.5)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out1 = self.measure(self.q_device)
                    param.copy_(param - np.pi)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out2 = self.measure(self.q_device)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                           self.hamil_info['hamil_list']],
                                          device=x.device).double()
        x_axis = []
        y_axis = []
        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
        for grad in self.grad_list:
            grad[x_axis, y_axis] = 0
        
        self.circuit_output = x
        self.circuit_output.requires_grad = True
        self.num_forwards += (1 + 2 * self.n_params) * x.shape[0]
        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def backprop_grad(self):
        for i, param in enumerate(self.parameters()):
            param.grad = torch.sum(self.grad_list[i] * self.circuit_output.grad).to(dtype=torch.float32).view(param.shape)

model_dict = {
    'vqe_0': QVQEModel0,
}
