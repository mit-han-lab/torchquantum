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

import pickle
import random
import sys
from copy import deepcopy

import torch
from qiskit import QuantumCircuit
from torchpack.utils.config import configs
from utils.circ_dag_converter import circ_to_dag_with_data


def load_data_from_raw(file_name):
    file = open("data/raw_data_qasm/" + file_name, "rb")
    data = pickle.load(file)
    file.close()
    print("Size of the data: ", len(data))
    return raw_pyg_converter(data)


def load_data_from_pyg(file_name):
    try:
        return load_normalized_data(file_name)
    except:
        try:
            file = open("data/pyg_data/" + file_name, "rb")
            normalize_data(file_name)
        except:
            load_data_and_save(file_name)
            normalize_data(file_name)
        return load_normalized_data(file_name)


def load_normalized_data(file_name):
    file = open("data/normalized_data/" + file_name, "rb")
    data = pickle.load(file)
    file.close()
    print("Size of the data: ", len(data))
    return data


def normalize_data(file_name):
    file = open("data/pyg_data/" + file_name, "rb")
    data = pickle.load(file)
    file.close()
    if configs.evalmode:
        file = open("data/normalized_data/" + configs.dataset.name + "meta", "rb")
        meta = pickle.load(file)
        file.close()
        print(meta)
        print(meta[1])
        for k, dag in enumerate(data):
            data[k].x = (dag.x - meta[0]) / (1e-8 + meta[1])
            data[k].global_features = (dag.global_features - meta[2]) / (1e-8 + meta[3])
    else:
        all_features = None
        for k, dag in enumerate(data):
            if not k:
                all_features = dag.x
                global_features = dag.global_features
                liu_features = dag.liu_features
            else:
                all_features = torch.cat([all_features, dag.x])
                global_features = torch.cat([global_features, dag.global_features])
                liu_features = torch.cat([liu_features, dag.liu_features])

        means = all_features.mean(0)
        stds = all_features.std(0)
        means_gf = global_features.mean(0)
        stds_gf = global_features.std(0)
        means_liu = liu_features.mean(0)
        stds_liu = liu_features.std(0)
        for k, dag in enumerate(data):
            data[k].x = (dag.x - means) / (1e-8 + stds)
            data[k].global_features = (dag.global_features - means_gf) / (
                1e-8 + stds_gf
            )
            data[k].liu_features = (dag.liu_features - means_liu) / (1e-8 + stds_liu)
        file = open("data/normalized_data/" + file_name + "meta", "wb")
        pickle.dump([means, stds, means_gf, stds_gf], file)
        file.close()
    file = open("data/normalized_data/" + file_name, "wb")
    pickle.dump(data, file)
    file.close()


def load_data_and_save(file_name):
    file = open("data/raw_data_qasm/" + file_name, "rb")
    data = pickle.load(file)
    file.close()
    pyg_data = raw_pyg_converter(data)
    random.shuffle(pyg_data)
    file = open("data/pyg_data/" + file_name, "wb")
    pickle.dump(pyg_data, file)
    file.close()


def raw_pyg_converter(dataset):
    pygdataset = []
    for data in dataset:
        circ = QuantumCircuit()
        circ = circ.from_qasm_str(data[0])
        dag = circ_to_dag_with_data(circ, data[1])
        dag.y = data[2]
        pygdataset.append(dag)
    return pygdataset


if __name__ == "__main__":
    file_name = sys.argv[1]
    dataset = load_data_and_save(file_name)
