import torchquantum as tq

from torchquantum.algorithm import VQE, Hamiltonian
from qiskit import QuantumCircuit

from torchquantum.plugin import qiskit2tq_op_history

if __name__ == "__main__":
    hamil = Hamiltonian.from_file("./examples/simple_vqe/h2.txt")

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
