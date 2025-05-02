# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: MIT

import math
import argparse

import torch
from torch import nn
from torchquantum.plugin.cuquantum import *
from torchquantum.operator.standard_gates import *





class MAXCUT(nn.Module):
    def __init__(self, n_wires, input_graph, n_layers):
        super().__init__()
        self.n_wires = n_wires
        self.input_graph = input_graph
        self.n_layers = n_layers

        self.circuit = ParameterizedQuantumCircuit(n_wires=n_wires, n_input_params=0, n_trainable_params=2 * n_layers)
        self.circuit.set_trainable_params(torch.randn(2 * n_layers))

        for wire in range(self.n_wires):
            self.circuit.append_gate(Hadamard, wires=wire)

        for l in range(self.n_layers):
            # mixer layer
            for i in range(self.n_wires):
                self.circuit.append_gate(RX, wires=i, trainable_idx=l)

            # entangler layer
            for edge in self.input_graph:
                self.circuit.append_gate(CNOT, wires=[edge[0], edge[1]])
                self.circuit.append_gate(RZ, wires=edge[1], trainable_idx=n_layers + l)
                self.circuit.append_gate(CNOT, wires=[edge[0], edge[1]])


        hamiltonian = {}
        for edge in self.input_graph:
            pauli_string = ""
            for wire in range(self.n_wires):
                if wire in edge:
                    pauli_string += "Z"
                else:
                    pauli_string += "I"
            hamiltonian[pauli_string] = 0.5

        backend = CuTensorNetworkBackend(TNConfig(num_hyper_samples=10))
        self.energy = QuantumExpectation(self.circuit, [hamiltonian], backend)
        self.sampling = QuantumSampling(self.circuit, 100, backend)

    def forward(self):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        output = self.energy() - len(self.input_graph) / 2
        end_time.record()

        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        print(f"Forward pass took {elapsed_time:.2f} ms")

        return output


def optimize(model, n_steps=100, lr=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"The initial parameters are:\n{next(model.parameters()).data.tolist()}")
    print("")
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = model()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        loss.backward()
        end_time.record()

        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        print(f"Backward pass took {elapsed_time:.2f} ms")

        optimizer.step()

        print(f"Step: {step}, Cost Objective: {loss.item()}")

    print("")
    print(f"The optimal parameters are:\n{next(model.parameters()).data.tolist()}")
    print("")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_wires", type=int, default=4, help="number of wires")
    parser.add_argument("--n_layers", type=int, default=4, help="number of layers")
    parser.add_argument("--steps", type=int, default=100, help="number of steps")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # create a fully connected graph
    input_graph = []
    for i in range(args.n_wires):
        for j in range(i):
            input_graph.append((i, j))

    print(f"Cost Objective Minimum (Analytic Reference Result): {math.floor(args.n_wires**2 // 4)}")

    model = MAXCUT(n_wires=args.n_wires, input_graph=input_graph, n_layers=args.n_layers)
    optimize(model, n_steps=args.steps, lr=args.lr)
    samples = model.sampling()

    print(f"Sampling Results: {samples}")


if __name__ == "__main__":
    main()
