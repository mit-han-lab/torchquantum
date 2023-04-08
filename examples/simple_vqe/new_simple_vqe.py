import torchquantum as tq
import torch
from torchquantum.vqe_utils import parse_hamiltonian_file
import random
import numpy as np
import argparse
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchquantum.measurement import expval_joint_analytical

from torchquantum.algorithms import VQE, Hamiltonian

if __name__ == "__main__":
    hamil = Hamiltonian.from_file("./h2.txt")

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
