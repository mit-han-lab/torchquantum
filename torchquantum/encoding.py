# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import argparse
#
# import torchquantum as tq
#
# from torch.optim.lr_scheduler import CosineAnnealingLR
#
# import random
# import numpy as np
#
# # data is cos(theta)|000> + e^(j * phi)sin(theta) |111>
#
# from torchpack.datasets.dataset import Dataset
# from torchquantum.plugins import tq2qiskit_initialize, tq2qiskit, tq2qiskit_measurement, qiskit_assemble_circs
#
# def gen_data(L, N):
#     omega_0 = np.zeros([2 ** L], dtype='complex_')
#     omega_0[0] = 1 + 0j
#
#     omega_1 = np.zeros([2 ** L], dtype='complex_')
#     omega_1[-1] = 1 + 0j
#
#     states = np.zeros([N, 2 ** L], dtype='complex_')
#
#     thetas = 2 * np.pi * np.random.rand(N)
#     phis = 2 * np.pi * np.random.rand(N)
#
#     for i in range(N):
#         states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
#
#     X = np.sin(2 * thetas) * np.cos(phis)
#
#     return states, X
#
#
# class RegressionDataset:
#     def __init__(self,
#                  split,
#                  n_samples,
#                  n_wires
#                  ):
#         self.split = split
#         self.n_samples = n_samples
#         self.n_wires = n_wires
#
#         self.states, self.Xlabel = gen_data(self.n_wires, self.n_samples)
#
#     def __getitem__(self, index: int):
#         instance = {'states': self.states[index],
#                     'Xlabel': self.Xlabel[index]}
#         return instance
#
#     def __len__(self) -> int:
#         return self.n_samples
#
#
# class Regression(Dataset):
#     def __init__(self, n_train, n_valid, n_wires):
#         n_samples_dict = {
#             'train': n_train,
#             'valid': n_valid
#         }
#         super().__init__({
#             split: RegressionDataset(
#                 split=split,
#                 n_samples=n_samples_dict[split],
#                 n_wires=n_wires
#             )
#             for split in ['train', 'valid']
#         })
#
#
# class QModel(tq.QuantumModule):
#     class QLayer(tq.QuantumModule):
#         def __init__(self, n_wires, n_blocks):
#             super().__init__()
#             # inside one block, we have one u3 layer one each qubit and one layer
#             # cu3 layer with ring connection
#             self.n_wires = n_wires
#             self.n_blocks = n_blocks
#             self.rx_layers = tq.QuantumModuleList()
#             self.ry_layers = tq.QuantumModuleList()
#             self.rz_layers = tq.QuantumModuleList()
#             self.cnot_layers = tq.QuantumModuleList()
#
#             for _ in range(n_blocks):
#                 self.rx_layers.append(tq.Op1QAllLayer(op=tq.RX,
#                                                       n_wires=n_wires,
#                                                       has_params=True,
#                                                       trainable=True,
#                                                       ))
#                 self.ry_layers.append(tq.Op1QAllLayer(op=tq.RY,
#                                                       n_wires=n_wires,
#                                                       has_params=True,
#                                                       trainable=True,
#                                                       ))
#                 self.rz_layers.append(tq.Op1QAllLayer(op=tq.RZ,
#                                                       n_wires=n_wires,
#                                                       has_params=True,
#                                                       trainable=True,
#                                                       ))
#                 self.cnot_layers.append(tq.Op2QAllLayer(op=tq.CNOT,
#                                                        n_wires=n_wires,
#                                                        has_params=False,
#                                                        trainable=False,
#                                                        circular=True
#                                                        ))
#
#         def forward(self, q_device: tq.QuantumDevice):
#             for k in range(self.n_blocks):
#                 self.rx_layers[k](q_device)
#                 self.ry_layers[k](q_device)
#                 self.rz_layers[k](q_device)
#                 self.cnot_layers[k](q_device)
#
#     def __init__(self, n_wires, n_blocks):
#         super().__init__()
#         self.q_layer = self.QLayer(n_wires=n_wires, n_blocks=n_blocks)
#         self.encoder = tq.StateEncoder()
#         self.measure = tq.MeasureAll(tq.PauliZ)
#
#     def forward(self, q_device: tq.QuantumDevice, input_states,
#                 use_qiskit=False):
#         self.q_device = q_device
#         # firstly set the q_device states
#         # q_device.set_states(input_states)
#         devi = input_states.device
#         if use_qiskit:
#             encoder_circs = tq2qiskit_initialize(q_device, input_states.detach().cpu().numpy())
#             q_layer_circ = tq2qiskit(q_device, self.q_layer)
#             measurement_circ = tq2qiskit_measurement(q_device,
#                                                      self.measure)
#             assembled_circs = qiskit_assemble_circs(encoder_circs,
#                                                     q_layer_circ,
#                                                     measurement_circ)
#             res = self.qiskit_processor.process_ready_circs(
#                 self.q_device, assembled_circs).to(devi)
#         else:
#             self.encoder(q_device, input_states)
#             self.q_layer(q_device)
#             res = self.measure(q_device)
#
#         return res
#
#
# def train(dataflow, q_device, model, device, optimizer, qiskit=False):
#     for feed_dict in dataflow['train']:
#         inputs = feed_dict['states'].to(device).to(torch.complex64)
#         targets = feed_dict['Xlabel'].to(device).to(torch.float)
#
#         outputs = model(q_device, inputs, qiskit)
#
#         loss = F.mse_loss(outputs[:, 1], targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f"loss: {loss.item()}")
#
#
# def valid_test(dataflow, q_device, split, model, device, qiskit):
#     target_all = []
#     output_all = []
#     with torch.no_grad():
#         for feed_dict in dataflow[split]:
#             inputs = feed_dict['states'].to(device).to(torch.complex64)
#             targets = feed_dict['Xlabel'].to(device).to(torch.float)
#
#             outputs = model(q_device, inputs, qiskit)
#
#             target_all.append(targets)
#             output_all.append(outputs)
#         target_all = torch.cat(target_all, dim=0)
#         output_all = torch.cat(output_all, dim=0)
#
#     loss = F.mse_loss(output_all[:, 1], target_all)
#
#     print(f"{split} set loss: {loss}")
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pdb', action='store_true', help='debug with pdb')
#     parser.add_argument('--bsz', type=int, default=32,
#                         help='batch size for training and validation')
#     parser.add_argument('--n_wires', type=int, default=3,
#                         help='number of qubits')
#     parser.add_argument('--n_blocks', type=int, default=2,
#                         help='number of blocks, each contain one layer of '
#                              'U3 gates and one layer of CU3 with '
#                              'ring connections')
#     parser.add_argument('--n_train', type=int, default=100,
#                         help='number of training samples')
#     parser.add_argument('--n_valid', type=int, default=100,
#                         help='number of validation samples')
#     parser.add_argument('--epochs', type=int, default=5,
#                         help='number of training epochs')
#
#     args = parser.parse_args()
#
#     if args.pdb:
#         import pdb
#         pdb.set_trace()
#
#     seed = 0
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     dataset = Regression(
#         n_train=args.n_train,
#         n_valid=args.n_valid,
#         n_wires=args.n_wires,
#     )
#
#     dataflow = dict()
#
#     for split in dataset:
#         if split == 'train':
#             sampler = torch.utils.data.RandomSampler(dataset[split])
#         else:
#             sampler = torch.utils.data.SequentialSampler(dataset[split])
#         dataflow[split] = torch.utils.data.DataLoader(
#             dataset[split],
#             batch_size=args.bsz,
#             sampler=sampler,
#             num_workers=1,
#             pin_memory=True)
#
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     model = QModel(n_wires=args.n_wires,
#                    n_blocks=args.n_blocks).to(device)
#
#     n_epochs = args.epochs
#     optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
#     scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
#
#     q_device = tq.QuantumDevice(n_wires=args.n_wires)
#     q_device.reset_states(bsz=args.bsz)
#
#     for epoch in range(1, n_epochs + 1):
#         # train
#         print(f"Epoch {epoch}, RL: {optimizer.param_groups[0]['lr']}")
#         train(dataflow, q_device, model, device, optimizer)
#
#         # valid
#         valid_test(dataflow, q_device, 'valid', model, device, False)
#         scheduler.step()
#
#     try:
#         from qiskit import IBMQ
#         from torchquantum.plugins import QiskitProcessor
#         print(f"\nTest with Qiskit Simulator")
#         processor_simulation = QiskitProcessor(use_real_qc=False)
#         model.set_qiskit_processor(processor_simulation)
#         valid_test(dataflow, q_device, 'test', model, device, qiskit=True)
#
#     except:
#         pass
#
#     # final valid
#     valid_test(dataflow, q_device, 'valid', model, device, True)
#
#
# if __name__ == '__main__':
#     main()

import torch
import torchquantum as tq

from torchquantum.functional import func_name_dict
from typing import Iterable
from torchquantum.macro import C_DTYPE
from abc import ABCMeta
from qiskit import QuantumCircuit


class Encoder(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, qdev: tq.QuantumDevice, x):
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
    def forward(self, qdev: tq.QuantumDevice, x):
        for info in self.func_list:
            if tq.op_name_dict[info["func"]].num_params > 0:
                params = x[:, info["input_idx"]]
            else:
                params = None
            func_name_dict[info["func"]](
                qdev,
                wires=info["wires"],
                params=params,
                static=self.static_mode,
                parent_graph=self.graph,
            )

    def to_qiskit(self, n_wires, x):
        # assuming the x is in batch mode
        bsz = x.shape[0]

        circs = []
        for k in range(bsz):
            circ = QuantumCircuit(n_wires)
            for info in self.func_list:
                if info["func"] == "rx":
                    circ.rx(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "ry":
                    circ.ry(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rz":
                    circ.rz(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rxx":
                    circ.rxx(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "ryy":
                    circ.ryy(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rzz":
                    circ.rzz(x[k][info["input_idx"][0]].item(), *info["wires"])
                elif info["func"] == "rzx":
                    circ.rzx(x[k][info["input_idx"][0]].item(), *info["wires"])
                else:
                    raise NotImplementedError(info["func"])
            circs.append(circ)

        return circs


class PhaseEncoder(Encoder, metaclass=ABCMeta):
    def __init__(self, func):
        super().__init__()
        self.func = func

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x):
        for k in range(qdev.n_wires):
            self.func(
                qdev,
                wires=k,
                params=x[:, k],
                static=self.static_mode,
                parent_graph=self.graph,
            )


class MultiPhaseEncoder(Encoder, metaclass=ABCMeta):
    def __init__(self, funcs, wires=None):
        super().__init__()
        self.funcs = funcs if isinstance(funcs, Iterable) else [funcs]
        self.wires = wires

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x):
        if self.wires is None:
            self.wires = list(range(qdev.n_wires)) * (len(self.funcs) // qdev.n_wires)

        x_id = 0
        for k, func in enumerate(self.funcs):
            if func in ["rx", "ry", "rz", "u1", "phaseshift"]:
                stride = 1
            elif func == "u2":
                stride = 2
            elif func == "u3":
                stride = 3
            else:
                raise ValueError(func)

            func_name_dict[func](
                qdev,
                wires=self.wires[k],
                params=x[:, x_id : (x_id + stride)],
                static=self.static_mode,
                parent_graph=self.graph,
            )
            x_id += stride


class StateEncoder(Encoder, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def forward(self, qdev: tq.QuantumDevice, x):
        # encoder the x to the statevector of the quantum device

        # normalize the input
        x = x / (torch.sqrt((x.abs() ** 2).sum(dim=-1))).unsqueeze(-1)
        state = torch.cat(
            (
                x,
                torch.zeros(
                    x.shape[0], 2**qdev.n_wires - x.shape[1], device=x.device
                ),
            ),
            dim=-1,
        )
        state = state.view([x.shape[0]] + [2] * qdev.n_wires)

        qdev.states = state.type(C_DTYPE)


class MagnitudeEncoder(Encoder, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()


AmplitudeEncoder = StateEncoder

encoder_op_list_name_dict = {
    "1x1_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
    ],
    "2x1_ryry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
    ],
    "3x1_ryryry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
    ],
    "3x1_rxrxrx": [
        {"input_idx": [0], "func": "rx", "wires": [0]},
        {"input_idx": [1], "func": "rx", "wires": [1]},
        {"input_idx": [2], "func": "rx", "wires": [2]},
    ],
    "4_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
    ],
    "4x4_u3rx": [
        {"input_idx": [0, 1, 2], "func": "u3", "wires": [0]},
        {"input_idx": [3], "func": "rx", "wires": [0]},
        {"input_idx": [4, 5, 6], "func": "u3", "wires": [1]},
        {"input_idx": [7], "func": "rx", "wires": [1]},
        {"input_idx": [8, 9, 10], "func": "u3", "wires": [2]},
        {"input_idx": [11], "func": "rx", "wires": [2]},
        {"input_idx": [12, 13, 14], "func": "u3", "wires": [3]},
        {"input_idx": [15], "func": "rx", "wires": [3]},
    ],
    "4x4_ryzxy": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "rz", "wires": [0]},
        {"input_idx": [5], "func": "rz", "wires": [1]},
        {"input_idx": [6], "func": "rz", "wires": [2]},
        {"input_idx": [7], "func": "rz", "wires": [3]},
        {"input_idx": [8], "func": "rx", "wires": [0]},
        {"input_idx": [9], "func": "rx", "wires": [1]},
        {"input_idx": [10], "func": "rx", "wires": [2]},
        {"input_idx": [11], "func": "rx", "wires": [3]},
        {"input_idx": [12], "func": "ry", "wires": [0]},
        {"input_idx": [13], "func": "ry", "wires": [1]},
        {"input_idx": [14], "func": "ry", "wires": [2]},
        {"input_idx": [15], "func": "ry", "wires": [3]},
    ],
    "16_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
    ],
    "4x4_rzsx": [
        {"input_idx": [0], "func": "rz", "wires": [0]},
        {"input_idx": None, "func": "sx", "wires": [0]},
        {"input_idx": [1], "func": "rz", "wires": [1]},
        {"input_idx": None, "func": "sx", "wires": [1]},
        {"input_idx": [2], "func": "rz", "wires": [2]},
        {"input_idx": None, "func": "sx", "wires": [2]},
        {"input_idx": [3], "func": "rz", "wires": [3]},
        {"input_idx": None, "func": "sx", "wires": [3]},
        {"input_idx": [4], "func": "rz", "wires": [0]},
        {"input_idx": None, "func": "sx", "wires": [0]},
        {"input_idx": [5], "func": "rz", "wires": [1]},
        {"input_idx": None, "func": "sx", "wires": [1]},
        {"input_idx": [6], "func": "rz", "wires": [2]},
        {"input_idx": None, "func": "sx", "wires": [2]},
        {"input_idx": [7], "func": "rz", "wires": [3]},
        {"input_idx": None, "func": "sx", "wires": [3]},
        {"input_idx": [8], "func": "rz", "wires": [0]},
        {"input_idx": None, "func": "sx", "wires": [0]},
        {"input_idx": [9], "func": "rz", "wires": [1]},
        {"input_idx": None, "func": "sx", "wires": [1]},
        {"input_idx": [10], "func": "rz", "wires": [2]},
        {"input_idx": None, "func": "sx", "wires": [2]},
        {"input_idx": [11], "func": "rz", "wires": [3]},
        {"input_idx": None, "func": "sx", "wires": [3]},
        {"input_idx": [12], "func": "rz", "wires": [0]},
        {"input_idx": None, "func": "sx", "wires": [0]},
        {"input_idx": [13], "func": "rz", "wires": [1]},
        {"input_idx": None, "func": "sx", "wires": [1]},
        {"input_idx": [14], "func": "rz", "wires": [2]},
        {"input_idx": None, "func": "sx", "wires": [2]},
        {"input_idx": [15], "func": "rz", "wires": [3]},
        {"input_idx": None, "func": "sx", "wires": [3]},
    ],
    "15_ryrz": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "rz", "wires": [0]},
    ],
    "2x8_ryzxyzxyz": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "rz", "wires": [0]},
        {"input_idx": [3], "func": "rz", "wires": [1]},
        {"input_idx": [4], "func": "rx", "wires": [0]},
        {"input_idx": [5], "func": "rx", "wires": [1]},
        {"input_idx": [6], "func": "ry", "wires": [0]},
        {"input_idx": [7], "func": "ry", "wires": [1]},
        {"input_idx": [8], "func": "rz", "wires": [0]},
        {"input_idx": [9], "func": "rz", "wires": [1]},
        {"input_idx": [10], "func": "rx", "wires": [0]},
        {"input_idx": [11], "func": "rx", "wires": [1]},
        {"input_idx": [12], "func": "ry", "wires": [0]},
        {"input_idx": [13], "func": "ry", "wires": [1]},
        {"input_idx": [14], "func": "rz", "wires": [0]},
        {"input_idx": [15], "func": "rz", "wires": [1]},
    ],
    "10_ryzx": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "rz", "wires": [0]},
        {"input_idx": [5], "func": "rz", "wires": [1]},
        {"input_idx": [6], "func": "rz", "wires": [2]},
        {"input_idx": [7], "func": "rz", "wires": [3]},
        {"input_idx": [8], "func": "rx", "wires": [0]},
        {"input_idx": [9], "func": "rx", "wires": [1]},
    ],
    "10_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
    ],
    "25_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
        {"input_idx": [16], "func": "ry", "wires": [16]},
        {"input_idx": [17], "func": "ry", "wires": [17]},
        {"input_idx": [18], "func": "ry", "wires": [18]},
        {"input_idx": [19], "func": "ry", "wires": [19]},
        {"input_idx": [20], "func": "ry", "wires": [20]},
        {"input_idx": [21], "func": "ry", "wires": [21]},
        {"input_idx": [22], "func": "ry", "wires": [22]},
        {"input_idx": [23], "func": "ry", "wires": [23]},
        {"input_idx": [24], "func": "ry", "wires": [24]},
    ],
    "25_ryrz": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
        {"input_idx": [16], "func": "ry", "wires": [16]},
        {"input_idx": [17], "func": "ry", "wires": [17]},
        {"input_idx": [18], "func": "ry", "wires": [18]},
        {"input_idx": [19], "func": "ry", "wires": [19]},
        {"input_idx": [20], "func": "ry", "wires": [20]},
        {"input_idx": [21], "func": "rz", "wires": [0]},
        {"input_idx": [22], "func": "rz", "wires": [1]},
        {"input_idx": [23], "func": "rz", "wires": [2]},
        {"input_idx": [24], "func": "rz", "wires": [3]},
    ],
    "6x6_ryzxy": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "rz", "wires": [0]},
        {"input_idx": [11], "func": "rz", "wires": [1]},
        {"input_idx": [12], "func": "rz", "wires": [2]},
        {"input_idx": [13], "func": "rz", "wires": [3]},
        {"input_idx": [14], "func": "rz", "wires": [4]},
        {"input_idx": [15], "func": "rz", "wires": [5]},
        {"input_idx": [16], "func": "rz", "wires": [6]},
        {"input_idx": [17], "func": "rz", "wires": [7]},
        {"input_idx": [18], "func": "rz", "wires": [8]},
        {"input_idx": [19], "func": "rz", "wires": [9]},
        {"input_idx": [20], "func": "rx", "wires": [0]},
        {"input_idx": [21], "func": "rx", "wires": [1]},
        {"input_idx": [22], "func": "rx", "wires": [2]},
        {"input_idx": [23], "func": "rx", "wires": [3]},
        {"input_idx": [24], "func": "rx", "wires": [4]},
        {"input_idx": [25], "func": "rx", "wires": [5]},
        {"input_idx": [26], "func": "rx", "wires": [6]},
        {"input_idx": [27], "func": "rx", "wires": [7]},
        {"input_idx": [28], "func": "rx", "wires": [8]},
        {"input_idx": [29], "func": "rx", "wires": [9]},
        {"input_idx": [30], "func": "ry", "wires": [0]},
        {"input_idx": [31], "func": "ry", "wires": [1]},
        {"input_idx": [32], "func": "ry", "wires": [2]},
        {"input_idx": [33], "func": "ry", "wires": [3]},
        {"input_idx": [34], "func": "ry", "wires": [4]},
        {"input_idx": [35], "func": "ry", "wires": [5]},
    ],
    "6x6_ryrz": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
        {"input_idx": [16], "func": "ry", "wires": [16]},
        {"input_idx": [17], "func": "ry", "wires": [17]},
        {"input_idx": [18], "func": "ry", "wires": [18]},
        {"input_idx": [19], "func": "ry", "wires": [19]},
        {"input_idx": [20], "func": "ry", "wires": [20]},
        {"input_idx": [21], "func": "rz", "wires": [0]},
        {"input_idx": [22], "func": "rz", "wires": [1]},
        {"input_idx": [23], "func": "rz", "wires": [2]},
        {"input_idx": [24], "func": "rz", "wires": [3]},
        {"input_idx": [25], "func": "rz", "wires": [4]},
        {"input_idx": [26], "func": "rz", "wires": [5]},
        {"input_idx": [27], "func": "rz", "wires": [6]},
        {"input_idx": [28], "func": "rz", "wires": [7]},
        {"input_idx": [29], "func": "rz", "wires": [8]},
        {"input_idx": [30], "func": "rz", "wires": [9]},
        {"input_idx": [31], "func": "rz", "wires": [10]},
        {"input_idx": [32], "func": "rz", "wires": [11]},
        {"input_idx": [33], "func": "rz", "wires": [12]},
        {"input_idx": [34], "func": "rz", "wires": [13]},
        {"input_idx": [35], "func": "rz", "wires": [14]},
    ],
    "6x6_ryrzrx": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
        {"input_idx": [16], "func": "rz", "wires": [0]},
        {"input_idx": [17], "func": "rz", "wires": [1]},
        {"input_idx": [18], "func": "rz", "wires": [2]},
        {"input_idx": [19], "func": "rz", "wires": [3]},
        {"input_idx": [20], "func": "rz", "wires": [4]},
        {"input_idx": [21], "func": "rz", "wires": [5]},
        {"input_idx": [22], "func": "rz", "wires": [6]},
        {"input_idx": [23], "func": "rz", "wires": [7]},
        {"input_idx": [24], "func": "rz", "wires": [8]},
        {"input_idx": [25], "func": "rz", "wires": [9]},
        {"input_idx": [26], "func": "rz", "wires": [10]},
        {"input_idx": [27], "func": "rz", "wires": [11]},
        {"input_idx": [28], "func": "rz", "wires": [12]},
        {"input_idx": [29], "func": "rz", "wires": [13]},
        {"input_idx": [30], "func": "rz", "wires": [14]},
        {"input_idx": [31], "func": "rz", "wires": [15]},
        {"input_idx": [32], "func": "rx", "wires": [0]},
        {"input_idx": [33], "func": "rx", "wires": [1]},
        {"input_idx": [34], "func": "rx", "wires": [2]},
        {"input_idx": [35], "func": "rx", "wires": [3]},
    ],
    "10x10_ryzxyzxyzxy": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "rz", "wires": [0]},
        {"input_idx": [11], "func": "rz", "wires": [1]},
        {"input_idx": [12], "func": "rz", "wires": [2]},
        {"input_idx": [13], "func": "rz", "wires": [3]},
        {"input_idx": [14], "func": "rz", "wires": [4]},
        {"input_idx": [15], "func": "rz", "wires": [5]},
        {"input_idx": [16], "func": "rz", "wires": [6]},
        {"input_idx": [17], "func": "rz", "wires": [7]},
        {"input_idx": [18], "func": "rz", "wires": [8]},
        {"input_idx": [19], "func": "rz", "wires": [9]},
        {"input_idx": [20], "func": "rx", "wires": [0]},
        {"input_idx": [21], "func": "rx", "wires": [1]},
        {"input_idx": [22], "func": "rx", "wires": [2]},
        {"input_idx": [23], "func": "rx", "wires": [3]},
        {"input_idx": [24], "func": "rx", "wires": [4]},
        {"input_idx": [25], "func": "rx", "wires": [5]},
        {"input_idx": [26], "func": "rx", "wires": [6]},
        {"input_idx": [27], "func": "rx", "wires": [7]},
        {"input_idx": [28], "func": "rx", "wires": [8]},
        {"input_idx": [29], "func": "rx", "wires": [9]},
        {"input_idx": [30], "func": "ry", "wires": [0]},
        {"input_idx": [31], "func": "ry", "wires": [1]},
        {"input_idx": [32], "func": "ry", "wires": [2]},
        {"input_idx": [33], "func": "ry", "wires": [3]},
        {"input_idx": [34], "func": "ry", "wires": [4]},
        {"input_idx": [35], "func": "ry", "wires": [5]},
        {"input_idx": [36], "func": "ry", "wires": [6]},
        {"input_idx": [37], "func": "ry", "wires": [7]},
        {"input_idx": [38], "func": "ry", "wires": [8]},
        {"input_idx": [39], "func": "ry", "wires": [9]},
        {"input_idx": [40], "func": "rz", "wires": [0]},
        {"input_idx": [41], "func": "rz", "wires": [1]},
        {"input_idx": [42], "func": "rz", "wires": [2]},
        {"input_idx": [43], "func": "rz", "wires": [3]},
        {"input_idx": [44], "func": "rz", "wires": [4]},
        {"input_idx": [45], "func": "rz", "wires": [5]},
        {"input_idx": [46], "func": "rz", "wires": [6]},
        {"input_idx": [47], "func": "rz", "wires": [7]},
        {"input_idx": [48], "func": "rz", "wires": [8]},
        {"input_idx": [49], "func": "rz", "wires": [9]},
        {"input_idx": [50], "func": "rx", "wires": [0]},
        {"input_idx": [51], "func": "rx", "wires": [1]},
        {"input_idx": [52], "func": "rx", "wires": [2]},
        {"input_idx": [53], "func": "rx", "wires": [3]},
        {"input_idx": [54], "func": "rx", "wires": [4]},
        {"input_idx": [55], "func": "rx", "wires": [5]},
        {"input_idx": [56], "func": "rx", "wires": [6]},
        {"input_idx": [57], "func": "rx", "wires": [7]},
        {"input_idx": [58], "func": "rx", "wires": [8]},
        {"input_idx": [59], "func": "rx", "wires": [9]},
        {"input_idx": [60], "func": "ry", "wires": [0]},
        {"input_idx": [61], "func": "ry", "wires": [1]},
        {"input_idx": [62], "func": "ry", "wires": [2]},
        {"input_idx": [63], "func": "ry", "wires": [3]},
        {"input_idx": [64], "func": "ry", "wires": [4]},
        {"input_idx": [65], "func": "ry", "wires": [5]},
        {"input_idx": [66], "func": "ry", "wires": [6]},
        {"input_idx": [67], "func": "ry", "wires": [7]},
        {"input_idx": [68], "func": "ry", "wires": [8]},
        {"input_idx": [69], "func": "ry", "wires": [9]},
        {"input_idx": [70], "func": "rz", "wires": [0]},
        {"input_idx": [71], "func": "rz", "wires": [1]},
        {"input_idx": [72], "func": "rz", "wires": [2]},
        {"input_idx": [73], "func": "rz", "wires": [3]},
        {"input_idx": [74], "func": "rz", "wires": [4]},
        {"input_idx": [75], "func": "rz", "wires": [5]},
        {"input_idx": [76], "func": "rz", "wires": [6]},
        {"input_idx": [77], "func": "rz", "wires": [7]},
        {"input_idx": [78], "func": "rz", "wires": [8]},
        {"input_idx": [79], "func": "rz", "wires": [9]},
        {"input_idx": [80], "func": "rx", "wires": [0]},
        {"input_idx": [81], "func": "rx", "wires": [1]},
        {"input_idx": [82], "func": "rx", "wires": [2]},
        {"input_idx": [83], "func": "rx", "wires": [3]},
        {"input_idx": [84], "func": "rx", "wires": [4]},
        {"input_idx": [85], "func": "rx", "wires": [5]},
        {"input_idx": [86], "func": "rx", "wires": [6]},
        {"input_idx": [87], "func": "rx", "wires": [7]},
        {"input_idx": [88], "func": "rx", "wires": [8]},
        {"input_idx": [89], "func": "rx", "wires": [9]},
        {"input_idx": [90], "func": "ry", "wires": [0]},
        {"input_idx": [91], "func": "ry", "wires": [1]},
        {"input_idx": [92], "func": "ry", "wires": [2]},
        {"input_idx": [93], "func": "ry", "wires": [3]},
        {"input_idx": [94], "func": "ry", "wires": [4]},
        {"input_idx": [95], "func": "ry", "wires": [5]},
        {"input_idx": [96], "func": "ry", "wires": [6]},
        {"input_idx": [97], "func": "ry", "wires": [7]},
        {"input_idx": [98], "func": "ry", "wires": [8]},
        {"input_idx": [99], "func": "ry", "wires": [9]},
    ],
    "8x8_ryzxyzxy": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "rz", "wires": [0]},
        {"input_idx": [11], "func": "rz", "wires": [1]},
        {"input_idx": [12], "func": "rz", "wires": [2]},
        {"input_idx": [13], "func": "rz", "wires": [3]},
        {"input_idx": [14], "func": "rz", "wires": [4]},
        {"input_idx": [15], "func": "rz", "wires": [5]},
        {"input_idx": [16], "func": "rz", "wires": [6]},
        {"input_idx": [17], "func": "rz", "wires": [7]},
        {"input_idx": [18], "func": "rz", "wires": [8]},
        {"input_idx": [19], "func": "rz", "wires": [9]},
        {"input_idx": [20], "func": "rx", "wires": [0]},
        {"input_idx": [21], "func": "rx", "wires": [1]},
        {"input_idx": [22], "func": "rx", "wires": [2]},
        {"input_idx": [23], "func": "rx", "wires": [3]},
        {"input_idx": [24], "func": "rx", "wires": [4]},
        {"input_idx": [25], "func": "rx", "wires": [5]},
        {"input_idx": [26], "func": "rx", "wires": [6]},
        {"input_idx": [27], "func": "rx", "wires": [7]},
        {"input_idx": [28], "func": "rx", "wires": [8]},
        {"input_idx": [29], "func": "rx", "wires": [9]},
        {"input_idx": [30], "func": "ry", "wires": [0]},
        {"input_idx": [31], "func": "ry", "wires": [1]},
        {"input_idx": [32], "func": "ry", "wires": [2]},
        {"input_idx": [33], "func": "ry", "wires": [3]},
        {"input_idx": [34], "func": "ry", "wires": [4]},
        {"input_idx": [35], "func": "ry", "wires": [5]},
        {"input_idx": [36], "func": "ry", "wires": [6]},
        {"input_idx": [37], "func": "ry", "wires": [7]},
        {"input_idx": [38], "func": "ry", "wires": [8]},
        {"input_idx": [39], "func": "ry", "wires": [9]},
        {"input_idx": [40], "func": "rz", "wires": [0]},
        {"input_idx": [41], "func": "rz", "wires": [1]},
        {"input_idx": [42], "func": "rz", "wires": [2]},
        {"input_idx": [43], "func": "rz", "wires": [3]},
        {"input_idx": [44], "func": "rz", "wires": [4]},
        {"input_idx": [45], "func": "rz", "wires": [5]},
        {"input_idx": [46], "func": "rz", "wires": [6]},
        {"input_idx": [47], "func": "rz", "wires": [7]},
        {"input_idx": [48], "func": "rz", "wires": [8]},
        {"input_idx": [49], "func": "rz", "wires": [9]},
        {"input_idx": [50], "func": "rx", "wires": [0]},
        {"input_idx": [51], "func": "rx", "wires": [1]},
        {"input_idx": [52], "func": "rx", "wires": [2]},
        {"input_idx": [53], "func": "rx", "wires": [3]},
        {"input_idx": [54], "func": "rx", "wires": [4]},
        {"input_idx": [55], "func": "rx", "wires": [5]},
        {"input_idx": [56], "func": "rx", "wires": [6]},
        {"input_idx": [57], "func": "rx", "wires": [7]},
        {"input_idx": [58], "func": "rx", "wires": [8]},
        {"input_idx": [59], "func": "rx", "wires": [9]},
        {"input_idx": [60], "func": "ry", "wires": [0]},
        {"input_idx": [61], "func": "ry", "wires": [1]},
        {"input_idx": [62], "func": "ry", "wires": [2]},
        {"input_idx": [63], "func": "ry", "wires": [3]},
    ],
    "8x2_ryz": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "rz", "wires": [0]},
        {"input_idx": [9], "func": "rz", "wires": [1]},
        {"input_idx": [10], "func": "rz", "wires": [2]},
        {"input_idx": [11], "func": "rz", "wires": [3]},
        {"input_idx": [12], "func": "rz", "wires": [4]},
        {"input_idx": [13], "func": "rz", "wires": [5]},
        {"input_idx": [14], "func": "rz", "wires": [6]},
        {"input_idx": [15], "func": "rz", "wires": [7]},
    ],
    "16x1_ry": [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [8]},
        {"input_idx": [9], "func": "ry", "wires": [9]},
        {"input_idx": [10], "func": "ry", "wires": [10]},
        {"input_idx": [11], "func": "ry", "wires": [11]},
        {"input_idx": [12], "func": "ry", "wires": [12]},
        {"input_idx": [13], "func": "ry", "wires": [13]},
        {"input_idx": [14], "func": "ry", "wires": [14]},
        {"input_idx": [15], "func": "ry", "wires": [15]},
    ],
}
