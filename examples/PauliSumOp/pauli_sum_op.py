# import torch
# import torchquantum as tq
# import torchquantum.functional as tqf
#
# import random
# import numpy as np
#
# from torchquantum.functional import mat_dict
#
# from torchquantum.plugins import tq2qiskit, qiskit2tq
# from torchquantum.measurement import expval_joint_analytical
# from torchquantum.plugins import op_history2qiskit
#
# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
#
# class MAXCUT(tq.QuantumModule):
#     """computes the optimal cut for a given graph.
#     outputs: the most probable bitstring decides the set {0 or 1} each
#     node belongs to.
#     """
#
#     def __init__(self, n_wires, input_graph, n_layers):
#         super().__init__()
#
#         self.n_wires = n_wires
#
#         self.input_graph = input_graph  # list of edges
#         self.n_layers = n_layers
#
#         self.betas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))
#         self.gammas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))
#
#     def mixer(self, qdev, beta):
#         """
#         Apply the single rotation and entangling layer of the QAOA ansatz.
#         mixer = exp(-i * beta * sigma_x)
#         """
#         for wire in range(self.n_wires):
#             qdev.rx(
#                 wires=wire,
#                 params=beta.unsqueeze(0),
#             ) # type: ignore
#
#     def entangler(self, qdev, gamma):
#         """
#         Apply the single rotation and entangling layer of the QAOA ansatz.
#         entangler = exp(-i * gamma * (1 - sigma_z * sigma_z)/2)
#         """
#         for edge in self.input_graph:
#             qdev.cx(
#                 [edge[0], edge[1]],
#             ) # type: ignore
#             qdev.rz(
#                 wires=edge[1],
#                 params=gamma.unsqueeze(0),
#             ) # type: ignore
#             qdev.cx(
#                 [edge[0], edge[1]],
#             ) # type: ignore
#
#     def edge_to_PauliString(self, edge):
#         # construct pauli string
#         pauli_string = ""
#         for wire in range(self.n_wires):
#             if wire in edge:
#                 pauli_string += "Z"
#             else:
#                 pauli_string += "I"
#         return pauli_string
#
#     def circuit(self, qdev):
#         """
#         execute the quantum circuit
#         """
#         # print(self.betas, self.gammas)
#         for wire in range(self.n_wires):
#             qdev.h(
#                 wires=wire,
#             ) # type: ignore
#
#         for i in range(self.n_layers):
#             self.mixer(qdev, self.betas[i])
#             self.entangler(qdev, self.gammas[i])
#
#     def forward(self, measure_all=False):
#         """
#         Apply the QAOA ansatz and only measure the edge qubit on z-basis.
#         Args:
#             if edge is None
#         """
#         qdev = tq.QuantumDevice(n_wires=self.n_wires, device=self.betas.device, record_op=False)
#
#         self.circuit(qdev)
#
#         # turn on the record_op above to print the circuit
#         # print(op_history2qiskit(self.n_wires, qdev.op_history))
#
#         # print(tq.measure(qdev, n_shots=1024))
#         # compute the expectation value
#         # print(qdev.get_states_1d())
#         if measure_all is False:
#             expVal = 0
#             for edge in self.input_graph:
#                 pauli_string = self.edge_to_PauliString(edge)
#                 expv = expval_joint_analytical(qdev, observable=pauli_string)
#                 expVal += 0.5 * expv
#                 # print(pauli_string, expv)
#             # print(expVal)
#             return expVal
#         else:
#             return tq.measure(qdev, n_shots=1024, draw_id=0)
#
# def backprop_optimize(model, n_steps=100, lr=0.1):
#     """
#     Optimize the QAOA ansatz over the parameters gamma and beta
#     Args:
#         betas (np.array): A list of beta parameters.
#         gammas (np.array): A list of gamma parameters.
#         n_steps (int): The number of steps to optimize, defaults to 10.
#         lr (float): The learning rate, defaults to 0.1.
#     """
#     # measure all edges in the input_graph
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     print(
#         "The initial parameters are betas = {} and gammas = {}".format(
#             *model.parameters()
#         )
#     )
#     # optimize the parameters and return the optimal values
#     for step in range(n_steps):
#         optimizer.zero_grad()
#         loss = model()
#         loss.backward()
#         optimizer.step()
#         if step % 2 == 0:
#             print("Step: {}, Cost Objective: {}".format(step, loss.item()))
#
#     print(
#         "The optimal parameters are betas = {} and gammas = {}".format(
#             *model.parameters()
#         )
#     )
#     return model(measure_all=True)
#
# def main():
#     # create a input_graph
#     input_graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
#     n_wires = 4
#     n_layers = 3
#     model = MAXCUT(n_wires=n_wires, input_graph=input_graph, n_layers=n_layers)
#     # model.to("cuda")
#     # model.to(torch.device("cuda"))
#     # circ = tq2qiskit(tq.QuantumDevice(n_wires=4), model)
#     # print(circ)
#     # print("The circuit is", circ.draw(output="mpl"))
#     # circ.draw(output="mpl")
#     # use backprop
#     backprop_optimize(model, n_steps=300, lr=0.01)
#     # use parameter shift rule
#     # param_shift_optimize(model, n_steps=500, step_size=100000)
#
# """
# Notes:
# 1. input_graph = [(0, 1), (3, 0), (1, 2), (2, 3)], mixer 1st & entangler 2nd, n_layers >= 2, answer is correct.
#
# """
#
# if __name__ == "__main__":
#     # import pdb
#     # pdb.set_trace()
#
#     main()

import torch
import argparse

import torchquantum as tq
import torchquantum.functional as tqf

import random
import numpy as np


class QLayer(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))

        # gates with trainable parameters
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

        self.measure = tq.MeasureMultiPauliSum(
            obs_list=[
                {
                    "wires": [0, 2, 3, 1],
                    "observables": ["x", "y", "z", "i"],
                    "coefficient": [1, 0.5, 0.4, 0.3],
                },
                {
                    "wires": [0, 2, 3, 1],
                    "observables": ["x", "x", "z", "i"],
                    "coefficient": [1, 0.5, 0.4, 0.3],
                },
            ]
        )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        """
        1. To convert tq QuantumModule to qiskit or run in the static
        model, need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard below
        """
        self.q_device = q_device

        self.random_layer(self.q_device)

        # some trainable gates (instantiated ahead of time)
        self.rx0(self.q_device, wires=0)
        self.ry0(self.q_device, wires=1)
        self.rz0(self.q_device, wires=3)
        self.crx0(self.q_device, wires=[0, 2])

        # add some more non-parameterized gates (add on-the-fly)
        tqf.hadamard(
            self.q_device, wires=3, static=self.static_mode, parent_graph=self.graph
        )
        tqf.sx(self.q_device, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(
            self.q_device,
            wires=[3, 0],
            static=self.static_mode,
            parent_graph=self.graph,
        )

        return self.measure(self.q_device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="debug with pdb")

    args = parser.parse_args()

    if args.pdb:
        import pdb

        pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    q_model = QLayer()

    q_device = tq.QuantumDevice(n_wires=4)
    q_device.reset_states(bsz=1)
    res = q_model(q_device)
    print(res)


if __name__ == "__main__":
    main()
