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
#                                   qiskit_assemble_circs)
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
#         def forward(self, q_device: tq.QuantumDevice):
#             """
#             1. To convert tq QuantumModule to qiskit or run in the static
#             model, need to:
#                 (1) add @tq.static_support before the forward
#                 (2) make sure to add
#                     static=self.static_mode and
#                     parent_graph=self.graph
#                     to all the tqf functions, such as tqf.hadamard below
#             """
#             self.q_device = q_device
#
#             self.random_layer(self.q_device)
#
#             # some trainable gates (instantiated ahead of time)
#             self.rx0(self.q_device, wires=0)
#             self.ry0(self.q_device, wires=1)
#             self.rz0(self.q_device, wires=3)
#             self.crx0(self.q_device, wires=[0, 2])
#
#             # add some more non-parameterized gates (add on-the-fly)
#             tqf.hadamard(self.q_device, wires=3, static=self.static_mode,
#                          parent_graph=self.graph)
#             tqf.sx(self.q_device, wires=2, static=self.static_mode,
#                    parent_graph=self.graph)
#             tqf.cnot(self.q_device, wires=[3, 0], static=self.static_mode,
#                      parent_graph=self.graph)
#             tqf.rx(self.q_device, wires=1, params=torch.tensor([0.1]),
#                    static=self.static_mode, parent_graph=self.graph)
#
#     def __init__(self):
#         super().__init__()
#         self.n_wires = 4
#         self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
#         self.encoder = tq.GeneralEncoder(
#             tq.encoder_op_list_name_dict['4x4_ryzxy'])
#
#         self.q_layer = self.QLayer()
#         self.measure = tq.MeasureAll(tq.PauliZ)
#
#     def forward(self, x, use_qiskit=False):
#         self.q_device.reset_states(x.shape[0])
#         bsz = x.shape[0]
#         x = F.avg_pool2d(x, 6).view(bsz, 16)
#         devi = x.device
#
#         if use_qiskit:
#             encoder_circs = tq2qiskit_expand_params(self.q_device, x,
#                                                     self.encoder.func_list)
#             q_layer_circ = tq2qiskit(self.q_device, self.q_layer)
#             measurement_circ = tq2qiskit_measurement(self.q_device,
#                                                      self.measure)
#             assembled_circs = qiskit_assemble_circs(encoder_circs,
#                                                     q_layer_circ,
#                                                     measurement_circ)
#             x0 = self.qiskit_processor.process_ready_circs(
#                 self.q_device, assembled_circs).to(devi)
#             # x1 = self.qiskit_processor.process_parameterized(
#             #     self.q_device, self.encoder, self.q_layer, self.measure, x)
#             # print((x0-x1).max())
#             x = x0
#
#         else:
#             self.encoder(self.q_device, x)
#             self.q_layer(self.q_device)
#             x = self.measure(self.q_device)
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

import torch
import torchquantum as tq
import torchquantum.functional as tqf

import random
import numpy as np

from torchquantum.functional import mat_dict

from torchquantum.plugins import tq2qiskit, qiskit2tq
from torchquantum.measurement import expval_joint_analytical

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class MAXCUT(tq.QuantumModule):
    """computes the optimal cut for a given graph.
    outputs: the most probable bitstring decides the set {0 or 1} each
    node belongs to.
    """

    def __init__(self, n_wires, input_graph, n_layers):
        super().__init__()

        self.n_wires = n_wires

        self.input_graph = input_graph  # list of edges
        self.n_layers = n_layers
        self.n_edges = len(input_graph)

        self.betas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))
        self.gammas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))

        self.reset_shift_param()

    def mixer(self, qdev, beta, layer_id):
        """
        Apply the single rotation and entangling layer of the QAOA ansatz.
        mixer = exp(-i * beta * sigma_x)
        """

        for wire in range(self.n_wires):
            if (
                self.shift_param_name == "beta"
                and self.shift_wire == wire
                and layer_id == self.shift_layer
            ):
                degree = self.shift_degree
            else:
                degree = 0
            qdev.rx(
                wires=wire,
                params=(beta.unsqueeze(0) + degree),
            )  # type: ignore

    def entangler(self, qdev, gamma, layer_id):
        """
        Apply the single rotation and entangling layer of the QAOA ansatz.
        entangler = exp(-i * gamma * (1 - sigma_z * sigma_z)/2)
        """
        for edge_id, edge in enumerate(self.input_graph):
            if (
                self.shift_param_name == "gamma"
                and edge_id == self.shift_edge_id
                and layer_id == self.shift_layer
            ):
                degree = self.shift_degree
            else:
                degree = 0
            qdev.cx(
                [edge[0], edge[1]],
            )  # type: ignore
            qdev.rz(
                wires=edge[1],
                params=(gamma.unsqueeze(0) + degree),
            )  # type: ignore
            qdev.cx(
                [edge[0], edge[1]],
            )  # type: ignore

    def set_shift_param(self, layer, wire, param_name, degree, edge_id):
        """
        set the shift parameter for the parameter shift rule
        """
        self.shift_layer = layer
        self.shift_wire = wire
        self.shift_param_name = param_name
        self.shift_degree = degree
        self.shift_edge_id = edge_id

    def reset_shift_param(self):
        """
        reset the shift parameter
        """
        self.shift_layer = None
        self.shift_wire = None
        self.shift_param_name = None
        self.shift_degree = None
        self.shift_edge_id = None

    def edge_to_PauliString(self, edge):
        # construct pauli string
        pauli_string = ""
        for wire in range(self.n_wires):
            if wire in edge:
                pauli_string += "Z"
            else:
                pauli_string += "I"
        return pauli_string

    def circuit(self, qdev):
        """
        execute the quantum circuit
        """
        # print(self.betas, self.gammas)
        for wire in range(self.n_wires):
            qdev.h(
                wires=wire,
            )  # type: ignore

        for i in range(self.n_layers):
            self.mixer(qdev, self.betas[i], i)
            self.entangler(qdev, self.gammas[i], i)

    def forward(self, measure_all=False):
        """
        Apply the QAOA ansatz and only measure the edge qubit on z-basis.
        Args:
            if edge is None
        """
        qdev = tq.QuantumDevice(n_wires=self.n_wires, device=self.betas.device)

        self.circuit(qdev)
        # print(tq.measure(qdev, n_shots=1024))
        # compute the expectation value
        # print(qdev.get_states_1d())
        if measure_all is False:
            expVal = 0
            for edge in self.input_graph:
                pauli_string = self.edge_to_PauliString(edge)
                expv = expval_joint_analytical(qdev, observable=pauli_string)
                expVal += 0.5 * expv
                # print(pauli_string, expv)
            # print(expVal)
            return expVal
        else:
            return tq.measure(qdev, n_shots=1024, draw_id=0)


def main():
    # create a input_graph
    input_graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
    n_wires = 4
    n_layers = 3
    model = MAXCUT(n_wires=n_wires, input_graph=input_graph, n_layers=n_layers)
    # model.to("cuda")
    # model.to(torch.device("cuda"))
    # circ = tq2qiskit(tq.QuantumDevice(n_wires=4), model)
    # print(circ)
    # print("The circuit is", circ.draw(output="mpl"))
    # circ.draw(output="mpl")
    # use backprop
    # backprop_optimize(model, n_steps=300, lr=0.01)
    # use parameter shift rule
    param_shift_optimize(model, n_steps=500, step_size=0.01)


def shift_and_run(model, use_qiskit=False):
    # flatten the parameters into 1D array

    grad_betas = []
    grad_gammas = []
    n_layers = model.n_layers
    n_wires = model.n_wires
    n_edges = model.n_edges

    for i in range(n_layers):
        grad_gamma = 0
        for k in range(n_edges):
            model.set_shift_param(i, None, "gamma", np.pi * 0.5, k)
            out1 = model(use_qiskit)
            model.reset_shift_param()

            model.set_shift_param(i, None, "gamma", -np.pi * 0.5, k)
            out2 = model(use_qiskit)
            model.reset_shift_param()

            grad_gamma += 0.5 * (out1 - out2).squeeze().item()
        grad_gammas.append(grad_gamma)

        grad_beta = 0
        for j in range(n_wires):
            model.set_shift_param(i, j, "beta", np.pi * 0.5, None)
            out1 = model(use_qiskit)
            model.reset_shift_param()

            model.set_shift_param(i, j, "beta", -np.pi * 0.5, None)
            out2 = model(use_qiskit)
            model.reset_shift_param()

            grad_beta += 0.5 * (out1 - out2).squeeze().item()
        grad_betas.append(grad_beta)

    return model(use_qiskit), [grad_betas, grad_gammas]


def param_shift_optimize(model, n_steps=10, step_size=0.1):
    """finds the optimal cut where parameter shift rule is used to compute the gradient"""
    # optimize the parameters and return the optimal values
    # print(
    # "The initial parameters are betas = {} and gammas = {}".format(
    # *model.parameters()
    # )
    # )
    n_layers = model.n_layers
    for step in range(n_steps):
        with torch.no_grad():
            loss, grad_list = shift_and_run(model)
        # param_list = list(model.parameters())
        # print(
        # "The initial parameters are betas = {} and gammas = {}".format(
        # *model.parameters()
        # )
        # )
        # param_list = torch.cat([param.flatten() for param in param_list])

        # print("The shape of the params", len(param_list), param_list[0].shape, param_list)
        # print("")
        # print("The shape of the grad_list = {}, 0th elem shape = {}, grad_list = {}".format(len(grad_list), grad_list[0].shape, grad_list))
        # print(grad_list, loss, model.betas, model.gammas)
        print(loss)
        with torch.no_grad():
            for i in range(n_layers):
                model.betas[i].copy_(model.betas[i] - step_size * grad_list[0][i])
                model.gammas[i].copy_(model.gammas[i] - step_size * grad_list[1][i])

            # for param, grad in zip(param_list, grad_list):
            # modify the parameters and ensure that there are no multiple views
            # param.copy_(param - step_size * grad)
        # if step % 5 == 0:
        # print("Step: {}, Cost Objective: {}".format(step, loss.item()))

        # print(
        #     "The updated parameters are betas = {} and gammas = {}".format(
        #         *model.parameters()
        #     )
        # )
    return model(measure_all=True)


"""
Notes:
1. input_graph = [(0, 1), (3, 0), (1, 2), (2, 3)], mixer 1st & entangler 2nd, n_layers >= 2, answer is correct.

"""

if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()

    main()
