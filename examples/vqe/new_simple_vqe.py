"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torchquantum as tq

from torchquantum.algorithm import VQE, Hamiltonian
from qiskit import QuantumCircuit

from torchquantum.plugin import qiskit2tq_op_history

if __name__ == "__main__":
    hamil = Hamiltonian.from_file("./examples/vqe/h2.txt")

    ops = [
        {'name': 'u3', 'wires': 0, 'trainable': True},
        {'name': 'u3', 'wires': 1, 'trainable': True},
        {'name': 'cu3', 'wires': [0, 1], 'trainable': True},
        {'name': 'cu3', 'wires': [1, 0], 'trainable': True},
        {'name': 'u3', 'wires': 0, 'trainable': True},
        {'name': 'u3', 'wires': 1, 'trainable': True},
        {'name': 'cu3', 'wires': [0, 1], 'trainable': True},
        {'name': 'cu3', 'wires': [1, 0], 'trainable': True},
    ]

    # or alternatively, you can use the following code to generate the ops
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.rx(0.1, 1)
    circ.cx(0, 1)
    circ.u(0.1, 0.2, 0.3, 0)
    circ.u(0.1, 0.2, 0.3, 0)
    circ.cx(1, 0)
    circ.u(0.1, 0.2, 0.3, 0)
    circ.u(0.1, 0.2, 0.3, 0)
    circ.cx(0, 1)
    circ.u(0.1, 0.2, 0.3, 0)
    circ.u(0.1, 0.2, 0.3, 0)
    circ.cx(1, 0)
    circ.u(0.1, 0.2, 0.3, 0)
    circ.u(0.1, 0.2, 0.3, 0)

    ops = qiskit2tq_op_history(circ)
    print(ops)

    ansatz = tq.QuantumModule.from_op_history(ops)
    configs = {
        "n_epochs": 10,
        "n_steps": 100,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "lr": 0.1,
        "device": "cuda",
    }
    vqe = VQE(
        hamil=hamil,
        ansatz=ansatz,
        train_configs=configs,
    )
    expval = vqe.train()
