import torch
import torchquantum as tq
import torchquantum.functional as tqf

import random
import numpy as np

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

        self.q_device = tq.QuantumDevice(n_wires=n_wires)

        self.rx0 = tq.RX(has_params=False, trainable=False)
        self.rz0 = tq.RZ(has_params=False, trainable=False)

        self.betas = torch.nn.Parameter(torch.rand(self.n_layers))
        self.gammas = torch.nn.Parameter(torch.rand(self.n_layers))

    def mixer_n_entangler(self, edge):
        """
        Apply the single rotation and entangling layer of the QAOA ansatz.
        mixer = exp(-i * beta * sigma_x)
        entangler = exp(-i * gamma * (1 - sigma_z * sigma_z)/2)
        """
        for wire in range(self.n_wires):
            for (beta, gamma) in zip(self.betas, self.gammas):
                # mixer
                self.rx0(self.q_device, wires=wire, params=2 * beta.unsqueeze(0))
                # entangler
                tqf.cx(self.q_device, [edge[0], edge[1]])
                self.rz0(self.q_device, wires=edge[1], params=2 * gamma.unsqueeze(0))
                tqf.cx(self.q_device, [edge[0], edge[1]])

    def circuit(self):
        """Run the QAOA circuit for the given edge.
        Args:
            edge (tuple): edge to be measured, defaults to None.
        Returns:
            the expectation value measured on the qubits of the edge.
        """
        tqf.h(self.q_device, wires=list(range(self.n_wires)))
        for k in range(self.n_layers):
            self.mixer_n_entangler(self.input_graph[k])

    def edge_to_PauliString(self, edge):
        # construct pauli string
        pauli_string = ""
        for wire in range(self.n_wires):
            if wire in edge:
                pauli_string += "Z"
            else:
                pauli_string += "I"
        return pauli_string

    def forward(self, measure_all=False):
        """
        Apply the QAOA ansatz and only measure the edge qubit on z-basis.
        Args:
            if edge is None
        """
        self.circuit()
        expVal = 0
        if measure_all is False:
            for edge in self.input_graph:
                pauli_string = self.edge_to_PauliString(edge)
                expVal += tq.expval_joint_analytical(
                    self.q_device, observable=pauli_string
                )
            return -expVal
        else:
            return tq.measure(self.q_device, n_shots=1024)


def optimize(model, n_steps=10, lr=0.1):
    """
    Optimize the QAOA ansatz over the parameters gamma and beta
    Args:
        betas (np.array): A list of beta parameters.
        gammas (np.array): A list of gamma parameters.
        n_steps (int): The number of steps to optimize, defaults to 10.
        lr (float): The learning rate, defaults to 0.1.
    """
    # measure all edges in the input_graph
    loss = model()
    print("The initial cost objective is {}".format(loss.item()))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(
        "The initial parameters are betas = {} and gammas = {}".format(
            *model.parameters()
        )
    )
    # optimize the parameters and return the optimal values
    for step in range(n_steps):
        optimizer.zero_grad()
        optimizer.step()
        if step % 2 == 0:
            print("Step: {}, Cost Objective: {}".format(step, loss.item()))

    print(
        "The optimal parameters are betas = {} and gammas = {}".format(
            *model.parameters()
        )
    )
    return model(measure_all=True)


def shift_and_run(model, use_qiskit=False):
    param_list = []
    for param in model.parameters():
        param_list.append(param)
    grad_list = []
    for param in param_list:
        param.copy_(param + np.pi * 0.5)
        out1 = model(use_qiskit)
        param.copy_(param - np.pi)
        out2 = model(use_qiskit)
        param.copy_(param + np.pi * 0.5)
        grad = 0.5 * (out1 - out2)
        grad_list.append(grad)
    return model(use_qiskit), grad_list


def main():
    # create a input_graph
    input_graph = [(0, 1), (3, 0), (1, 2), (2, 3)]
    n_wires = 4
    n_layers = 1
    # create a QAOA ansatz
    model = MAXCUT(n_wires=n_wires, input_graph=input_graph, n_layers=n_layers)
    # optimizer
    optimize(model, n_steps=10, lr=0.1)


if __name__ == "__main__":
    main()
