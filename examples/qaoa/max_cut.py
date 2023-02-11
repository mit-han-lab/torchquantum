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
    def __init__(self, n_wires, input_graph):
        super().__init__()
        self.n_wires = n_wires
        self.input_graph = input_graph  # list of edges
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def mixer(self, beta):
        """
        Apply the single rotation layer of the QAOA ansatz.
        mixer = exp(-i * beta * sigma_x)
        """
        for wire in range(self.n_wires):
            self.rx0.func(self.q_device, wires=wire, params=2 * beta)

    def entangler(self, gamma):
        """
        Apply the entangling layer of the QAOA ansatz.
        entangler = exp(-i * gamma * (1 - sigma_z * sigma_z)/2)
        """
        for edge in self.input_graph:
            tqf.cx(self.q_device, [edge[0], edge[1]])
            # self.rz0(self.q_device, wires = edge[1])
            self.rz0.func(self.q_device, params=gamma, wires=edge[1])
            tqf.cx(self.q_device, [edge[0], edge[1]])

    def forward(self, betas, gammas, edge=None):
        """
        Apply the QAOA ansatz and only measure the edge qubit on z-basis.

        Args:
            betas (np.array): A list of beta parameters.
            gammas (np.array): A list of gamma parameters.
            n_layers (int): The number of layers in the QAOA circuit, defaults to 1.
            edge (tuple of two ints): The edge to be measured, defaults to None.
        """
        # create a uniform superposition over all qubits
        tqf.h(self.q_device, list(range(self.n_wires)))
        n_layers = len(betas)

        for k in range(n_layers):
            self.mixer(betas[k].unsqueeze(0))
            self.entangler(gammas[k].unsqueeze(0))

        if edge is None:
            # if no edge is specified, measure all qubits
            return self.measure(self.q_device).flatten()

        # with only one shot calculate the expectation value of the edge qubit
        return tq.expval(
            self.q_device, wires=[*edge], observables=[tq.PauliZ(), tq.PauliZ()]
        )

    def optimize(self, betas, gammas, n_steps=10, lr=0.1, scheduler=None):
        """
        Optimize the QAOA ansatz over the parameters gamma and beta

        Args:
            betas (np.array): A list of beta parameters.
            gammas (np.array): A list of gamma parameters.
            n_steps (int): The number of steps to optimize, defaults to 100.
            lr (float): The learning rate, defaults to 0.1.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler, defaults to None.
        """

        # initialize the parameters randomly near zero
        n_layers = len(betas)
        betas = torch.rand(n_layers, requires_grad=True)
        gammas = torch.rand(n_layers, requires_grad=True)

        # measure all edges in the input_graph
        def cost_objective(betas, gammas):
            loss = 0
            for edge in self.input_graph:
                loss -= 1 / 2 * (1 - self.forward(betas, gammas, edge))
            return torch.mean(loss)

        loss = cost_objective(betas, gammas)

        # define the optimizer
        optimizer = torch.optim.Adam([betas, gammas], lr=lr)

        # optimize the parameters and return the optimal values
        for step in range(n_steps):
            optimizer.zero_grad()
            loss = cost_objective(betas, gammas)
            loss.backward(retain_graph=True)
            optimizer.step()
            # scheduler.step()
            print("Step: {}, Cost Objective: {}".format(step, loss.item()))

        # compute the bitsring with the highest probability
        expectation_vals = self.forward(betas, gammas, edge=None)
        # calculate the probability of zero and one from the expectation values
        probs_zero = (1 + expectation_vals) / 2
        probs_one = (1 - expectation_vals) / 2
        # find the outcome with the highest probability from probs_zero and probs_one
        outcome = np.where(probs_zero >= probs_one, 0, 1)
        bitstring = "".join(str(int(item)) for item in outcome)
        print("The bitstring with the highest probability is: {}".format(bitstring))
        print(
            "The optimal parameters are betas = {} and gammas = {}".format(
                betas, gammas
            )
        )
        return (betas, gammas)


def main():
    # create a input_graph
    input_graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
    n_wires = 4

    # create a QAOA ansatz
    qaoa = MAXCUT(n_wires, input_graph)

    # initialize the parameters randomly near zero
    betas = 0.01 * torch.rand(1, requires_grad=True)
    gammas = 0.01 * torch.rand(1, requires_grad=True)

    # test the mixer function
    print("Testing the mixer function...")
    for i in range(len(betas)):
        print(qaoa.entangler(betas[i].unsqueeze(0)))

    # test the entangler function
    print("Testing the entangler function...")
    for i in range(len(gammas)):
        print(qaoa.entangler(gammas[i].unsqueeze(0)))

    # test the forward function
    print("Testing the forward function...")
    for i in range(len(betas)):
        print(qaoa.forward(betas[i].unsqueeze(0), gammas[i].unsqueeze(0)))

    # #optimize the parameters
    print("Optimizing the parameters...")
    qaoa.optimize(betas, gammas)


if __name__ == "__main__":
    main()
