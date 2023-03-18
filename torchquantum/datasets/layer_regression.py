# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import argparse
#
# import torchquantum as tq
# import torchquantum.functional as tqf
#
# from torchquantum.plugins import (tq2qiskit_expand_params,
#                                   tq2qiskit,
#                                   tq2qiskit_measurement,
#                                   qiskit_assemble_circs,
#                                   op_history2qiskit,
#                                   op_history2qiskit_expand_params)
#
# from torchquantum.datasets import MNIST
# from torch.optim.lr_scheduler import CosineAnnealingLR
#
# import random
# import numpy as np
#
#
# class QFCModel(tq.QuantumModule):
#     class QLayer(tq.QuantumModule):
#         def __init__(self):
#             super().__init__()
#             self.n_wires = 4
#             self.random_layer = tq.RandomLayer(n_ops=50,
#                                                wires=list(range(self.n_wires)))
#
#             # gates with trainable parameters
#             self.rx0 = tq.RX(has_params=True, trainable=True)
#             self.ry0 = tq.RY(has_params=True, trainable=True)
#             self.rz0 = tq.RZ(has_params=True, trainable=True)
#             self.crx0 = tq.CRX(has_params=True, trainable=True)
#
#         @tq.static_support
#         def forward(self, qdev: tq.QuantumDevice):
#             """
#             1. To convert tq QuantumModule to qiskit or run in the static
#             model, need to:
#                 (1) add @tq.static_support before the forward
#                 (2) make sure to add
#                     static=self.static_mode and
#                     parent_graph=self.graph
#                     to all the tqf functions, such as tqf.hadamard below
#             """
#             self.random_layer(qdev)
#
#             # some trainable gates (instantiated ahead of time)
#             self.rx0(qdev, wires=0)
#             self.ry0(qdev, wires=1)
#             self.rz0(qdev, wires=3)
#             self.crx0(qdev, wires=[0, 2])
#
#             # add some more non-parameterized gates (add on-the-fly)
#             qdev.h(wires=3, static=self.static_mode, parent_graph=self.graph)
#             qdev.sx(wires=2, static=self.static_mode, parent_graph=self.graph)
#             qdev.cnot(wires=[3, 0], static=self.static_mode, parent_graph=self.graph)
#             qdev.rx(wires=1, params=torch.tensor([0.1]), static=self.static_mode, parent_graph=self.graph)
#
#     def __init__(self):
#         super().__init__()
#         self.n_wires = 4
#         self.encoder = tq.GeneralEncoder(
#             tq.encoder_op_list_name_dict['4x4_u3rx'])
#
#         self.q_layer = self.QLayer()
#         self.measure = tq.MeasureAll(tq.PauliZ)
#
#     def forward(self, x, use_qiskit=False):
#         qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True)
#
#         bsz = x.shape[0]
#         x = F.avg_pool2d(x, 6).view(bsz, 16)
#         devi = x.device
#
#         if use_qiskit:
#             encoder_circs = tq2qiskit_expand_params(qdev, x,
#                                                     self.encoder.func_list)
#             q_layer_circ = tq2qiskit(qdev, self.q_layer)
#             measurement_circ = tq2qiskit_measurement(qdev,
#                                                      self.measure)
#             assembled_circs = qiskit_assemble_circs(encoder_circs,
#                                                     q_layer_circ,
#                                                     measurement_circ)
#             x0 = self.qiskit_processor.process_ready_circs(
#                 qdev, assembled_circs).to(devi)
#             x = x0
#
#         else:
#             self.encoder(qdev, x)
#             op_history_parameterized = qdev.op_history
#             qdev.reset_op_history()
#             self.q_layer(qdev)
#             op_history_fixed = qdev.op_history
#             x = self.measure(qdev)
#
#         # circs = op_history2qiskit_expand_params(self.n_wires, op_history_parameterized, bsz=bsz)
#         # print(op_history2qiskit(self.n_wires, op_history_fixed))
#
#         x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
#         x = F.log_softmax(x, dim=1)
#
#         return x
#
#
# def train(dataflow, model, device, optimizer):
#     for feed_dict in dataflow['train']:
#         inputs = feed_dict['image'].to(device)
#         targets = feed_dict['digit'].to(device)
#
#         outputs = model(inputs)
#         loss = F.nll_loss(outputs, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f"loss: {loss.item()}", end='\r')
#
#
# def valid_test(dataflow, split, model, device, qiskit=False):
#     target_all = []
#     output_all = []
#     with torch.no_grad():
#         for feed_dict in dataflow[split]:
#             inputs = feed_dict['image'].to(device)
#             targets = feed_dict['digit'].to(device)
#
#             outputs = model(inputs, use_qiskit=qiskit)
#
#             target_all.append(targets)
#             output_all.append(outputs)
#         target_all = torch.cat(target_all, dim=0)
#         output_all = torch.cat(output_all, dim=0)
#
#     _, indices = output_all.topk(1, dim=1)
#     masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
#     size = target_all.shape[0]
#     corrects = masks.sum().item()
#     accuracy = corrects / size
#     loss = F.nll_loss(output_all, target_all).item()
#
#     print(f"{split} set accuracy: {accuracy}")
#     print(f"{split} set loss: {loss}")
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--static', action='store_true', help='compute with '
#                                                               'static mode')
#     parser.add_argument('--pdb', action='store_true', help='debug with pdb')
#     parser.add_argument('--wires-per-block', type=int, default=2,
#                         help='wires per block int static mode')
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
#     dataset = MNIST(
#         root='./mnist_data',
#         train_valid_split_ratio=[0.9, 0.1],
#         digits_of_interest=[3, 6],
#         n_test_samples=75,
#     )
#     dataflow = dict()
#
#     for split in dataset:
#         sampler = torch.utils.data.RandomSampler(dataset[split])
#         dataflow[split] = torch.utils.data.DataLoader(
#             dataset[split],
#             batch_size=256,
#             sampler=sampler,
#             num_workers=8,
#             pin_memory=True)
#
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     model = QFCModel().to(device)
#
#     n_epochs = args.epochs
#     optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
#     scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
#
#     if args.static:
#         # optionally to switch to the static mode, which can bring speedup
#         # on training
#         model.q_layer.static_on(wires_per_block=args.wires_per_block)
#
#     for epoch in range(1, n_epochs + 1):
#         # train
#         print(f"Epoch {epoch}:")
#         train(dataflow, model, device, optimizer)
#         print(optimizer.param_groups[0]['lr'])
#
#         # valid
#         valid_test(dataflow, 'valid', model, device)
#         scheduler.step()
#
#     # test
#     valid_test(dataflow, 'test', model, device, qiskit=False)
#
#     # run on Qiskit simulator and real Quantum Computers
#     try:
#         from qiskit import IBMQ
#         from torchquantum.plugins import QiskitProcessor
#
#         # firstly perform simulate
#         print(f"\nTest with Qiskit Simulator")
#         processor_simulation = QiskitProcessor(use_real_qc=False)
#         model.set_qiskit_processor(processor_simulation)
#         valid_test(dataflow, 'test', model, device, qiskit=True)
#
#         # then try to run on REAL QC
#         backend_name = 'ibmq_lima'
#         print(f"\nTest on Real Quantum Computer {backend_name}")
#         # Please specify your own hub group and project if you have the
#         # IBMQ premium plan to access more machines.
#         processor_real_qc = QiskitProcessor(use_real_qc=True,
#                                             backend_name=backend_name,
#                                             hub='ibm-q',
#                                             group='open',
#                                             project='main',
#                                             )
#         model.set_qiskit_processor(processor_real_qc)
#         valid_test(dataflow, 'test', model, device, qiskit=True)
#     except ImportError:
#         print("Please install qiskit, create an IBM Q Experience Account and "
#               "save the account token according to the instruction at "
#               "'https://github.com/Qiskit/qiskit-ibmq-provider', "
#               "then try again.")
#
#
# if __name__ == '__main__':
#     main()

from torchpack.datasets.dataset import Dataset
import numpy as np
import torch

__all__ = ["LayerRegression"]


class LayerRegressionDatasetV:
    def __init__(self):
        self.data = None
        self.n_instance = 10000
        self.input = np.random.rand(5).astype(np.float) * 2 * np.pi
        self.output = np.random.rand(5).astype(np.float) * 2 - 1

    def __getitem__(self, index: int):
        instance = {"input": self.input, "output": self.output}
        return instance

    def __len__(self) -> int:
        return self.n_instance


class LayerRegressionV(Dataset):
    def __init__(
        self,
    ):
        super().__init__(
            {split: LayerRegressionDatasetV() for split in ["train", "valid", "test"]}
        )


class LayerRegressionDataset:
    def __init__(self):
        self.data = None
        self.n_instance = 10000
        mat = torch.randn((2**5, 2**5), dtype=torch.complex64)
        mat = mat.svd()[0].data

        self.output = mat

    def __getitem__(self, index: int):
        instance = {"input": self.output, "output": self.output}
        return instance

    def __len__(self) -> int:
        return self.n_instance


class LayerRegression(Dataset):
    def __init__(
        self,
    ):
        super().__init__(
            {split: LayerRegressionDataset() for split in ["train", "valid", "test"]}
        )
