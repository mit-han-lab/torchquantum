# # pylint: disable=line-too-long
# from qiskit.algorithms import VQE
# from qiskit_nature.algorithms import (GroundStateEigensolver,
#                                       NumPyMinimumEigensolverFactory)
# from qiskit_nature.drivers import Molecule
# from qiskit_nature.drivers.second_quantization import (
#     ElectronicStructureMoleculeDriver, ElectronicStructureDriverType)
# from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
# from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
# from qiskit_nature.converters.second_quantization import QubitConverter
# from qiskit_nature.mappers.second_quantization import ParityMapper
# # pylint: enable=line-too-long
#
# import matplotlib.pyplot as plt
# import numpy as np
# from qiskit_nature.circuit.library import UCCSD, HartreeFock
# from qiskit.circuit.library import EfficientSU2
# from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP
# from qiskit.opflow import TwoQubitReduction
# from qiskit import BasicAer, Aer
# from qiskit.utils import QuantumInstance
# from qiskit.utils.mitigation import CompleteMeasFitter
# from qiskit.providers.aer.noise import NoiseModel
#
# import qiskit_nature
#
# qiskit_nature.settings.dict_aux_operators = False
#
#
#
# import pdb
# pdb.set_trace()
#
# def get_qubit_op(dist):
#     # Define Molecule
#     molecule = Molecule(
#         # Coordinates in Angstrom
#         geometry=[
#             ["Li", [0.0, 0.0, 0.0]],
#             ["H", [dist, 0.0, 0.0]]
#         ],
#         multiplicity=1,  # = 2*spin + 1
#         charge=0,
#     )
#
#     driver = ElectronicStructureMoleculeDriver(
#         molecule=molecule,
#         basis="sto3g",
#         driver_type=ElectronicStructureDriverType.PYSCF)
#
#     # Get properties
#     properties = driver.run()
#     num_particles = (properties
#                         .get_property("ParticleNumber")
#                         .num_particles)
#     num_spin_orbitals = int(properties
#                             .get_property("ParticleNumber")
#                             .num_spin_orbitals)
#
#     # Define Problem, Use freeze core approximation, remove orbitals.
#     problem = ElectronicStructureProblem(
#         driver,
#         [FreezeCoreTransformer(freeze_core=True,
#                                remove_orbitals=[-3,-2])])
#
#     second_q_ops = problem.second_q_ops()  # Get 2nd Quant OP
#     num_spin_orbitals = problem.num_spin_orbitals
#     num_particles = problem.num_particles
#
#     mapper = ParityMapper()  # Set Mapper
#     hamiltonian = second_q_ops[0]  # Set Hamiltonian
#     # Do two qubit reduction
#     converter = QubitConverter(mapper,two_qubit_reduction=True)
#     reducer = TwoQubitReduction(num_particles)
#     qubit_op = converter.convert(hamiltonian)
#     qubit_op = reducer.convert(qubit_op)
#
#     return qubit_op, num_particles, num_spin_orbitals, problem, converter
#
#
# def exact_solver(problem, converter):
#     solver = NumPyMinimumEigensolverFactory()
#     calc = GroundStateEigensolver(converter, solver)
#     result = calc.solve(problem)
#     return result
#
#
# backend = BasicAer.get_backend("statevector_simulator")
# distances = np.arange(0.5, 4.0, 0.2)
# exact_energies = []
# vqe_energies = []
# optimizer = SLSQP(maxiter=5)
#
# # pylint: disable=undefined-loop-variable
# for dist in distances:
#     (qubit_op, num_particles, num_spin_orbitals,
#                              problem, converter) = get_qubit_op(dist)
#     result = exact_solver(problem,converter)
#     exact_energies.append(result.total_energies[0].real)
#     init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
#     var_form = UCCSD(converter,
#                      num_particles,
#                      num_spin_orbitals,
#                      initial_state=init_state)
#     vqe = VQE(var_form, optimizer, quantum_instance=backend)
#     vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
#     vqe_result = problem.interpret(vqe_calc).total_energies[0].real
#     vqe_energies.append(vqe_result)
#     print(f"Interatomic Distance: {np.round(dist, 2)}",
#           f"VQE Result: {vqe_result:.5f}",
#           f"Exact Energy: {exact_energies[-1]:.5f}")
#
# print("All energies have been calculated")

from copy import deepcopy

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torchpack.utils.config import configs

from core.datasets import builder
from core.datasets.drawer import draw_curve, draw_scatter


class trainer:
    def __init__(self, model, device, criterion, optimizer, scheduler, loaders):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loaders = loaders
        self.best = 1e10
        self.best_params = None
        self.training_data = {}

    def train(self):
        self.model.train()
        self.training_data["train_loss"] = []
        self.training_data["val_error"] = []

        for epoch in range(configs.num_epochs):
            loss_sum = 0
            for batch in self.loaders["train"]:
                self.optimizer.zero_grad()
                out = self.model(batch)
                loss = self.criterion(out, batch.y.to(self.device))
                loss.backward()
                self.optimizer.step()
                loss_sum += (
                    loss.item() * len(batch.y) / len(self.loaders["train"].dataset)
                )
            self.scheduler.step()
            print(
                f"[{epoch + 1} / {configs.num_epochs}],sqrtloss={loss_sum**0.5} \r",
                end="",
            )
            self.training_data["train_loss"].append(loss_sum**0.5)
            if epoch % 5 == 0:
                val_error = self.valid()
                self.save_best(val_error)
                self.training_data["val_error"].append(val_error)
        torch.save(self.best_params, f"exp/{configs.exp_name}/model.pth")
        print("\n")

    def save_best(self, loss):
        if loss < self.best:
            self.best = loss
            self.best_params = deepcopy(self.model.state_dict())

    def valid(self):
        self.model.eval()
        pred_error = 0
        for batch in self.loaders["valid"]:
            out = self.model(batch)
            pred_error += ((out - batch.y.to(self.device)) ** 2).sum().item()
        pred_error = (pred_error / len(self.loaders["valid"].dataset)) ** 0.5
        print(
            f"\t\t\t\t\t\t val_error:{pred_error} \r",
            end="",
        )
        return pred_error

    def saveall(self):
        mydict = {}
        mydict["train_loss"] = self.training_data["train_loss"]
        mydict["val_error"] = self.training_data["val_error"]
        mydict["test_pred"] = self.training_data["test_pred"]
        mydict["test_y"] = self.training_data["test_y"]
        mydict["test_error"] = self.test_error
        mydict["best"] = self.best

        torch.save(mydict, f"exp/{configs.exp_name}/all.pth")

    def loadall(self):
        mydict = torch.load(f"exp/{configs.exp_name}/all.pth")
        self.training_data["train_loss"] = mydict["train_loss"]
        self.training_data["val_error"] = mydict["val_error"]
        self.training_data["test_pred"] = mydict["test_pred"]
        self.training_data["test_y"] = mydict["test_y"]
        self.test_error = mydict["test_error"]
        self.best = mydict["best"]
        print(f"test_error:{self.test_error}")
        print(f"best_val_error:{self.best}")

    def test(self):
        self.training_data["test_pred"] = np.array([])
        self.training_data["test_y"] = np.array([])
        self.test_error = 0
        print(len(self.loaders["test"].dataset))
        if len(self.loaders["test"].dataset) > 1:
            self.model.load_state_dict(torch.load(f"exp/{configs.exp_name}/model.pth"))
            self.model.eval()
            test_error = 0
            for batch in self.loaders["test"]:
                out = self.model(batch)
                self.training_data["test_pred"] = np.concatenate(
                    (self.training_data["test_pred"], out.cpu().detach().numpy())
                )
                self.training_data["test_y"] = np.concatenate(
                    (self.training_data["test_y"], batch.y.cpu().detach().numpy())
                )
                test_error += ((out - batch.y.to(self.device)) ** 2).sum().item()
            test_error = (test_error / len(self.loaders["test"].dataset)) ** 0.5
            self.test_error = test_error

    def testwith(self, dataname, device):
        self.training_data["test_pred_with" + dataname] = np.array([])
        self.training_data["test_y_with" + dataname] = np.array([])
        testdataset = builder.make_dataset_from(dataname).get_data(device, "test")
        testdataset = DataLoader(testdataset, batch_size=configs.batch_size)

        self.model.load_state_dict(torch.load(f"exp/{configs.exp_name}/model.pth"))
        self.model.eval()
        test_error = 0
        for batch in testdataset:
            out = self.model(batch)
            self.training_data["test_pred_with" + dataname] = np.concatenate(
                (
                    self.training_data["test_pred_with" + dataname],
                    out.cpu().detach().numpy(),
                )
            )
            self.training_data["test_y_with" + dataname] = np.concatenate(
                (
                    self.training_data["test_y_with" + dataname],
                    batch.y.cpu().detach().numpy(),
                )
            )
            test_error += ((out - batch.y.to(self.device)) ** 2).sum().item()
        test_error = (test_error / len(testdataset.dataset)) ** 0.5
        self.training_data["test_error_with" + dataname] = test_error
        print(f"test_error:{self.test_error}")
        # draw_scatter(
        #     self.training_data["test_y_with"],
        #     self.training_data["test_pred_with"],
        #     name="test.png",
        # )

    def save_training_data(self):
        torch.save(self.training_data, f"exp/{configs.exp_name}/training_data.pth")

    def scatter(self):
        draw_scatter(
            self.training_data["test_y"],
            self.training_data["test_pred"],
        )

    def curve(self):
        draw_curve(
            np.arange(len(self.training_data["train_loss"])),
            self.training_data["train_loss"],
        )
