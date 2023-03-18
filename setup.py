# import torchquantum as tq
# import torch
# from torchquantum.vqe_utils import parse_hamiltonian_file
# import random
# import numpy as np
# import argparse
# import torch.optim as optim
#
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchquantum.measurement import expval_joint_analytical
#
#
# class QVQEModel(tq.QuantumModule):
#     def __init__(self, arch, hamil_info):
#         super().__init__()
#         self.arch = arch
#         self.hamil_info = hamil_info
#         self.n_wires = hamil_info['n_wires']
#         self.n_blocks = arch['n_blocks']
#         self.u3_layers = tq.QuantumModuleList()
#         self.cu3_layers = tq.QuantumModuleList()
#         for _ in range(self.n_blocks):
#             self.u3_layers.append(tq.Op1QAllLayer(op=tq.U3,
#                                                   n_wires=self.n_wires,
#                                                   has_params=True,
#                                                   trainable=True,
#                                                   ))
#             self.cu3_layers.append(tq.Op2QAllLayer(op=tq.CU3,
#                                                    n_wires=self.n_wires,
#                                                    has_params=True,
#                                                    trainable=True,
#                                                    circular=True
#                                                    ))
#
#     def forward(self):
#         qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=next(self.parameters()).device)
#
#         for k in range(self.n_blocks):
#             self.u3_layers[k](qdev)
#             self.cu3_layers[k](qdev)
#
#         expval = 0
#         for hamil in self.hamil_info['hamil_list']:
#             expval += expval_joint_analytical(qdev, observable=hamil["pauli_string"]) * hamil["coeff"]
#
#         return expval
#
#
# def train(model, optimizer, n_steps=1):
#     for _ in range(n_steps):
#         loss = model()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f"Expectation of energy: {loss.item()}")
#
#
# def valid_test(model):
#     with torch.no_grad():
#         loss = model()
#
#     print(f"validation: expectation of energy: {loss.item()}")
#
#
# def process_hamil_info(hamil_info):
#     hamil_list = hamil_info['hamil_list']
#     n_wires = hamil_info["n_wires"]
#     all_info = []
#
#     for hamil in hamil_list:
#         pauli_string = ""
#         for i in range(n_wires):
#             if i in hamil['wires']:
#                 wire = hamil['wires'].index(i)
#                 pauli_string += (hamil['observables'][wire].upper())
#             else:
#                 pauli_string += "I"
#         all_info.append({"pauli_string": pauli_string,
#                             "coeff": hamil['coefficient']})
#     hamil_info['hamil_list'] = all_info
#     return hamil_info
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--pdb', action='store_true', help='debug with pdb')
#     parser.add_argument('--n_blocks', type=int, default=2,
#                         help='number of blocks, each contain one layer of '
#                              'U3 gates and one layer of CU3 with '
#                              'ring connections')
#     parser.add_argument('--steps_per_epoch', type=int, default=10,
#                         help='number of training epochs')
#     parser.add_argument('--epochs', type=int, default=100,
#                         help='number of training epochs')
#     parser.add_argument('--hamil_filename', type=str, default='./h2.txt',
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
#     hamil_info = process_hamil_info(parse_hamiltonian_file(args.hamil_filename))
#
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     model = QVQEModel(arch={"n_blocks": args.n_blocks}, hamil_info=hamil_info)
#
#     model.to(device)
#
#     n_epochs = args.epochs
#     optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
#     scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
#
#     for epoch in range(1, n_epochs + 1):
#         # train
#         print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
#         train(model, optimizer, n_steps=args.steps_per_epoch)
#
#         scheduler.step()
#
#     # final valid
#     valid_test(model)
#
#
# if __name__ == '__main__':
#     main()

from setuptools import setup, find_packages

VERSION = {}  # type: ignore

with open("torchquantum/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)

if __name__ == "__main__":
    setup(
        name="torchquantum",
        version=VERSION["version"],
        description="A PyTorch-based framework for differentiable classical simulation of quantum computing",
        url="https://github.com/mit-han-lab/torchquantum",
        author="Hanrui Wang, Jiannan Cao, Jessica Ding, Jiai Gu, Song Han, Zirui Li, Zhiding Liang, Pengyu Liu, Mohammadreza Tavasoli",
        author_email="hanruiwang.hw@gmail.com",
        license="MIT",
        install_requires=[
            "numpy>=1.19.2",
            "torchvision>=0.9.0.dev20210130",
            "tqdm>=4.56.0",
            "setuptools>=52.0.0",
            "torch>=1.8.0",
            "torchpack>=0.3.0",
            "qiskit==0.38.0",
            "matplotlib>=3.3.2",
            "pathos>=0.2.7",
            "pylatexenc>=2.10",
            "dill==0.3.4",
        ],
        extras_require={"doc": ["nbsphinx", "recommonmark"]},
        python_requires=">=3.5",
        include_package_data=True,
        packages=find_packages(),
    )
