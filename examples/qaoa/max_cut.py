import torch
import torchquantum as tq
import torchquantum.functional as tqf

import random
import numpy as np

from torchquantum.functional import mat_dict

# from torchquantum.plugins import tq2qiskit, qiskit2tq

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def expval_joint_analytical(
    q_device: tq.QuantumDevice,
    observable: str,
):
    """
    Compute the expectation value of a joint observable in analytical way, assuming the
    statevector is available.
    Args:
        q_device: the quantum device
        observable: the joint observable, on the qubit 0, 1, 2, 3, etc in this order
    Returns:
        the expectation value
    Examples:
    >>> import torchquantum as tq
    >>> import torchquantum.functional as tqf
    >>> x = tq.QuantumDevice(n_wires=2)
    >>> tqf.hadamard(x, wires=0)
    >>> tqf.x(x, wires=1)
    >>> tqf.cnot(x, wires=[0, 1])
    >>> print(expval_joint_analytical(x, 'II'))
    tensor([[1.0000]])
    >>> print(expval_joint_analytical(x, 'XX'))
    tensor([[1.0000]])
    >>> print(expval_joint_analytical(x, 'ZZ'))
    tensor([[-1.0000]])
    """
    # compute the hamiltonian matrix
    paulix = mat_dict["paulix"]
    pauliy = mat_dict["pauliy"]
    pauliz = mat_dict["pauliz"]
    iden = mat_dict["i"]
    pauli_dict = {"X": paulix, "Y": pauliy, "Z": pauliz, "I": iden}

    observable = observable.upper()
    assert len(observable) == q_device.n_wires
    hamiltonian = pauli_dict[observable[0]]
    for op in observable[1:]:
        hamiltonian = torch.kron(hamiltonian, pauli_dict[op])

    states = q_device.get_states_1d()

    return torch.mm(states, torch.mm(hamiltonian, states.conj().transpose(0, 1))).real


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

        self.betas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))
        self.gammas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))

    def mixer(self, beta):
        """
        Apply the single rotation and entangling layer of the QAOA ansatz.
        mixer = exp(-i * beta * sigma_x)
        """
        for wire in range(self.n_wires):
            tqf.rx(
                self.q_device,
                wires=wire,
                params=2 * beta.unsqueeze(0),
                static=self.static_mode,
                parent_graph=self.graph,
            )

    def entangler(self, gamma):
        """
        Apply the single rotation and entangling layer of the QAOA ansatz.
        entangler = exp(-i * gamma * (1 - sigma_z * sigma_z)/2)
        """
        for edge in self.input_graph:
            tqf.cx(
                self.q_device,
                [edge[0], edge[1]],
                static=self.static_mode,
                parent_graph=self.graph,
            )
            tqf.rz(
                self.q_device,
                wires=edge[1],
                params=gamma.unsqueeze(0),
                static=self.static_mode,
                parent_graph=self.graph,
            )
            tqf.cx(
                self.q_device,
                [edge[0], edge[1]],
                static=self.static_mode,
                parent_graph=self.graph,
            )

    def edge_to_PauliString(self, edge):
        # construct pauli string
        pauli_string = ""
        for wire in range(self.n_wires):
            if wire in edge:
                pauli_string += "Z"
            else:
                pauli_string += "I"
        return pauli_string

    def circuit(self):
        """
        execute the quantum circuit
        """
        tqf.h(
            self.q_device,
            wires=list(range(self.n_wires)),
            static=self.static_mode,
            parent_graph=self.graph,
        )

        for i in range(self.n_layers):
            self.entangler(self.gammas[i])
            self.mixer(self.betas[i])

    @tq.static_support
    def forward(self, measure_all=False):
        """
        Apply the QAOA ansatz and only measure the edge qubit on z-basis.
        Args:
            if edge is None
        """
        self.q_device.reset_states(1)
        self.circuit()
        # states = self.q_device.get_states_1d()
        # print(tq.measure(self.q_device, n_shots=1024))
        # compute the expectation value
        if measure_all is False:
            expVal = 0
            for edge in self.input_graph:
                pauli_string = self.edge_to_PauliString(edge)
                expVal -= 0.5 * (
                    1 - expval_joint_analytical(self.q_device, observable=pauli_string)
                )
            return expVal
        else:
            return tq.measure(self.q_device, n_shots=1024, draw_id=0)


def backprop_optimize(model, n_steps=100, lr=0.1):
    """
    Optimize the QAOA ansatz over the parameters gamma and beta
    Args:
        betas (np.array): A list of beta parameters.
        gammas (np.array): A list of gamma parameters.
        n_steps (int): The number of steps to optimize, defaults to 10.
        lr (float): The learning rate, defaults to 0.1.
    """
    # measure all edges in the input_graph
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(
        "The initial parameters are betas = {} and gammas = {}".format(
            *model.parameters()
        )
    )
    # optimize the parameters and return the optimal values
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
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

    # flatten the parameters into 1D array
    params_list = list(model.parameters())
    params_flat = torch.cat([param.flatten() for param in params_list])

    grad_list = []
    for param in params_flat:
        param.copy_(param + np.pi * 0.5)
        out1 = model(use_qiskit)
        param.copy_(param - np.pi)
        out2 = model(use_qiskit)
        # return the parameters to their original values
        param.copy_(param + np.pi * 0.5)
        grad = 0.5 * (out1 - out2)
        grad_list.append(grad)

    grad_flat = torch.tensor(grad_list)
    # unflatten the parameters
    return model(use_qiskit), grad_flat


def param_shift_optimize(model, n_steps=10, step_size=0.1):
    """finds the optimal cut where parameter shift rule is used to compute the gradient"""
    # optimize the parameters and return the optimal values
    print(
        "The initial parameters are betas = {} and gammas = {}".format(
            *model.parameters()
        )
    )

    for step in range(n_steps):
        with torch.no_grad():
            loss, grad_list = shift_and_run(model)
        param_list = list(model.parameters())
        print(
            "The initial parameters are betas = {} and gammas = {}".format(
                *model.parameters()
            )
        )
        param_list = torch.cat([param.flatten() for param in param_list])

        # print("The shape of the params", len(param_list), param_list[0].shape, param_list)
        # print("")
        # print("The shape of the grad_list = {}, 0th elem shape = {}, grad_list = {}".format(len(grad_list), grad_list[0].shape, grad_list))
        for param, grad in zip(param_list, grad_list):
            # modify the parameters and ensure that there are no multiple views
            param.copy_(param - step_size * grad)
        if step % 5 == 0:
            print("Step: {}, Cost Objective: {}".format(step, loss.item()))

        print(
            "The updated parameters are betas = {} and gammas = {}".format(
                *model.parameters()
            )
        )
    return model(measure_all=True)


def main():
    # create a input_graph
    input_graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
    n_wires = 4
    n_layers = 1
    model = MAXCUT(n_wires=n_wires, input_graph=input_graph, n_layers=n_layers)
    # circ = tq2qiskit(tq.QuantumDevice(n_wires=4), model)
    # print("The circuit is", circ.draw(output="mpl"))
    # circ.draw(output="mpl")
    # use backprop
    backprop_optimize(model, n_steps=30, lr=0.1)
    # use parameter shift rule
    # param_shift_optimize(model, n_steps=10, step_size=0.1)


"""
Notes:
1. input_graph = [(0, 1), (3, 0), (1, 2), (2, 3)], mixer 1st & entangler 2nd, n_layers =2, answer = correct.
                                                                              n_layers =1, answer = wrong, same probs for all edges.
2. input_graph = [(0, 1), (3, 0), (1, 2), (2, 3)], mixer 2nd & entangler 1st, n_layers =2, answer = correct.
                                                                              n_layers =1, answer = wrong, same probs for all edges.

"""

if __name__ == "__main__":
    main()
