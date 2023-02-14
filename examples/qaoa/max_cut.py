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
    outputs: the most probable bitstring decides the set {0 or 1} each node belongs to.
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

    def circuit(self, edge=None):
        """Run the QAOA circuit for the given edge.

        Args:
            edge (tuple): edge to be measured, defaults to None.

        Returns:
            the expectation value measured on the qubits corresponding to the edge.
        """
        self.q_device.reset_states(1)
        tqf.h(self.q_device, wires=list(range(self.n_wires)))

        for k in range(self.n_layers):
            self.mixer_n_entangler(self.input_graph[k])

        if edge is None:
            return tq.measure(self.q_device, n_shots=1024)

        exp_val = torch.prod(
            tq.expval(
                self.q_device, wires=[*edge], observables=[tq.PauliZ(), tq.PauliZ()]
            )
        )

        return exp_val

    def forward(self):
        """
        Apply the QAOA ansatz and only measure the edge qubit on z-basis.

        Args:
            betas (np.array): A list of beta parameters.
            gammas (np.array): A list of gamma parameters.
            n_layers (int): The number of layers in the QAOA circuit, defaults to 1.
            edge (tuple of two ints): The edge to be measured, defaults to None.
        """
        # create a uniform superposition over all qubits
        loss = 0
        for edge in self.input_graph:
            loss -= 1 / 2 * (1 - self.circuit(edge))
        return loss


def optimize(model, n_steps=10, lr=0.1):
    """
    Optimize the QAOA ansatz over the parameters gamma and beta

    Args:
        betas (np.array): A list of beta parameters.
        gammas (np.array): A list of gamma parameters.
        n_steps (int): The number of steps to optimize, defaults to 10.
        lr (float): The learning rate, defaults to 0.1.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler, defaults to None.
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
        optimizer.zero_grad()  # right now the optimizer goes forwards and backwards, so need to specify the edge to measure
        loss.backward(retain_graph=True)
        optimizer.step()
        if step % 2 == 0:
            print("Step: {}, Cost Objective: {}".format(step, loss.item()))

    print(
        "The optimal parameters are betas = {} and gammas = {}".format(
            *model.parameters()
        )
    )
    return model.circuit()


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
