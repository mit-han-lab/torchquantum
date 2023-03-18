# import torch
# import torchquantum as tq
# import torchquantum.functional as tqf
#
# import random
# import numpy as np
#
# from torchquantum.functional import mat_dict
#
# from torchquantum.plugins import tq2qiskit, qiskit2tq
# from torchquantum.measurement import expval_joint_analytical
# from torchquantum.plugins import op_history2qiskit
#
# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
#
# class MAXCUT(tq.QuantumModule):
#     """computes the optimal cut for a given graph.
#     outputs: the most probable bitstring decides the set {0 or 1} each
#     node belongs to.
#     """
#
#     def __init__(self, n_wires, input_graph, n_layers):
#         super().__init__()
#
#         self.n_wires = n_wires
#
#         self.input_graph = input_graph  # list of edges
#         self.n_layers = n_layers
#
#         self.betas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))
#         self.gammas = torch.nn.Parameter(0.01 * torch.rand(self.n_layers))
#
#     def mixer(self, qdev, beta):
#         """
#         Apply the single rotation and entangling layer of the QAOA ansatz.
#         mixer = exp(-i * beta * sigma_x)
#         """
#         for wire in range(self.n_wires):
#             qdev.rx(
#                 wires=wire,
#                 params=beta.unsqueeze(0),
#             ) # type: ignore
#
#     def entangler(self, qdev, gamma):
#         """
#         Apply the single rotation and entangling layer of the QAOA ansatz.
#         entangler = exp(-i * gamma * (1 - sigma_z * sigma_z)/2)
#         """
#         for edge in self.input_graph:
#             qdev.cx(
#                 [edge[0], edge[1]],
#             ) # type: ignore
#             qdev.rz(
#                 wires=edge[1],
#                 params=gamma.unsqueeze(0),
#             ) # type: ignore
#             qdev.cx(
#                 [edge[0], edge[1]],
#             ) # type: ignore
#
#     def edge_to_PauliString(self, edge):
#         # construct pauli string
#         pauli_string = ""
#         for wire in range(self.n_wires):
#             if wire in edge:
#                 pauli_string += "Z"
#             else:
#                 pauli_string += "I"
#         return pauli_string
#
#     def circuit(self, qdev):
#         """
#         execute the quantum circuit
#         """
#         # print(self.betas, self.gammas)
#         for wire in range(self.n_wires):
#             qdev.h(
#                 wires=wire,
#             ) # type: ignore
#
#         for i in range(self.n_layers):
#             self.mixer(qdev, self.betas[i])
#             self.entangler(qdev, self.gammas[i])
#
#     def forward(self, measure_all=False):
#         """
#         Apply the QAOA ansatz and only measure the edge qubit on z-basis.
#         Args:
#             if edge is None
#         """
#         qdev = tq.QuantumDevice(n_wires=self.n_wires, device=self.betas.device, record_op=False)
#
#         self.circuit(qdev)
#
#         # turn on the record_op above to print the circuit
#         # print(op_history2qiskit(self.n_wires, qdev.op_history))
#
#         # print(tq.measure(qdev, n_shots=1024))
#         # compute the expectation value
#         # print(qdev.get_states_1d())
#         if measure_all is False:
#             expVal = 0
#             for edge in self.input_graph:
#                 pauli_string = self.edge_to_PauliString(edge)
#                 expv = expval_joint_analytical(qdev, observable=pauli_string)
#                 expVal += 0.5 * expv
#                 # print(pauli_string, expv)
#             # print(expVal)
#             return expVal
#         else:
#             return tq.measure(qdev, n_shots=1024, draw_id=0)
#
# def backprop_optimize(model, n_steps=100, lr=0.1):
#     """
#     Optimize the QAOA ansatz over the parameters gamma and beta
#     Args:
#         betas (np.array): A list of beta parameters.
#         gammas (np.array): A list of gamma parameters.
#         n_steps (int): The number of steps to optimize, defaults to 10.
#         lr (float): The learning rate, defaults to 0.1.
#     """
#     # measure all edges in the input_graph
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     print(
#         "The initial parameters are betas = {} and gammas = {}".format(
#             *model.parameters()
#         )
#     )
#     # optimize the parameters and return the optimal values
#     for step in range(n_steps):
#         optimizer.zero_grad()
#         loss = model()
#         loss.backward()
#         optimizer.step()
#         if step % 2 == 0:
#             print("Step: {}, Cost Objective: {}".format(step, loss.item()))
#
#     print(
#         "The optimal parameters are betas = {} and gammas = {}".format(
#             *model.parameters()
#         )
#     )
#     return model(measure_all=True)
#
# def main():
#     # create a input_graph
#     input_graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
#     n_wires = 4
#     n_layers = 3
#     model = MAXCUT(n_wires=n_wires, input_graph=input_graph, n_layers=n_layers)
#     # model.to("cuda")
#     # model.to(torch.device("cuda"))
#     # circ = tq2qiskit(tq.QuantumDevice(n_wires=4), model)
#     # print(circ)
#     # print("The circuit is", circ.draw(output="mpl"))
#     # circ.draw(output="mpl")
#     # use backprop
#     backprop_optimize(model, n_steps=300, lr=0.01)
#     # use parameter shift rule
#     # param_shift_optimize(model, n_steps=500, step_size=100000)
#
# """
# Notes:
# 1. input_graph = [(0, 1), (3, 0), (1, 2), (2, 3)], mixer 1st & entangler 2nd, n_layers >= 2, answer is correct.
#
# """
#
# if __name__ == "__main__":
#     # import pdb
#     # pdb.set_trace()
#
#     main()

import argparse
import pdb
import torch
import torchquantum as tq
import numpy as np

from qiskit import Aer, execute
from torchpack.utils.logging import logger
from torchquantum.utils import (
    switch_little_big_endian_matrix,
    switch_little_big_endian_state,
    get_expectations_from_counts,
    find_global_phase,
)
from test.static_mode_test import QLayer as AllRandomLayer
from torchquantum.plugins import tq2qiskit
from torchquantum.macro import F_DTYPE


def unitary_tq_vs_qiskit_test():
    for n_wires in range(2, 10):
        q_dev = tq.QuantumDevice(n_wires=n_wires)
        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_layer = AllRandomLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=500,
            n_ops_cin=500,
            n_funcs=500,
            qiskit_compatible=True,
        )

        unitary_tq = q_layer.get_unitary(q_dev, x)
        unitary_tq = switch_little_big_endian_matrix(unitary_tq.data.numpy())

        # qiskit
        circ = tq2qiskit(q_layer, x)
        simulator = Aer.get_backend("unitary_simulator")
        result = execute(circ, simulator).result()
        unitary_qiskit = result.get_unitary(circ)

        stable_threshold = 1e-5
        try:
            # WARNING: need to remove the global phase! The qiskit simulated
            # results sometimes has global phase shift.
            global_phase = find_global_phase(
                unitary_tq, unitary_qiskit, stable_threshold
            )

            if global_phase is None:
                logger.exception(
                    f"Cannot find a stable enough factor to "
                    f"reduce the global phase, increase the "
                    f"stable_threshold and try again"
                )
                raise RuntimeError

            assert np.allclose(unitary_tq * global_phase, unitary_qiskit, atol=1e-6)
            logger.info(f"PASS tq vs qiskit [n_wires]={n_wires}")

        except AssertionError:
            logger.exception(f"FAIL tq vs qiskit [n_wires]={n_wires}")
            raise AssertionError

        except RuntimeError:
            raise RuntimeError

    logger.info(f"PASS tq vs qiskit unitary test")


def state_tq_vs_qiskit_test():
    bsz = 1
    for n_wires in range(2, 10):
        q_dev = tq.QuantumDevice(n_wires=n_wires)
        q_dev.reset_states(bsz=bsz)

        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_layer = AllRandomLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=500,
            n_ops_cin=500,
            n_funcs=500,
            qiskit_compatible=True,
        )

        q_layer(q_dev, x)
        state_tq = q_dev.states.reshape(bsz, -1)
        state_tq = switch_little_big_endian_state(state_tq.data.numpy())

        # qiskit
        circ = tq2qiskit(q_layer, x)
        # Select the StatevectorSimulator from the Aer provider
        simulator = Aer.get_backend("statevector_simulator")

        # Execute and get counts
        result = execute(circ, simulator).result()
        state_qiskit = result.get_statevector(circ)

        stable_threshold = 1e-5
        try:
            # WARNING: need to remove the global phase! The qiskit simulated
            # results sometimes has global phase shift.
            global_phase = find_global_phase(
                state_tq, np.expand_dims(state_qiskit, 0), stable_threshold
            )

            if global_phase is None:
                logger.exception(
                    f"Cannot find a stable enough factor to "
                    f"reduce the global phase, increase the "
                    f"stable_threshold and try again"
                )
                raise RuntimeError

            assert np.allclose(state_tq * global_phase, state_qiskit, atol=1e-6)
            logger.info(f"PASS tq vs qiskit [n_wires]={n_wires}")

        except AssertionError:
            logger.exception(f"FAIL tq vs qiskit [n_wires]={n_wires}")
            raise AssertionError

        except RuntimeError:
            raise RuntimeError

    logger.info(f"PASS tq vs qiskit statevector test")


def measurement_tq_vs_qiskit_test():
    bsz = 1
    for n_wires in range(2, 10):
        q_dev = tq.QuantumDevice(n_wires=n_wires)
        q_dev.reset_states(bsz=bsz)

        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_layer = AllRandomLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=500,
            n_ops_cin=500,
            n_funcs=500,
            qiskit_compatible=True,
        )

        q_layer(q_dev, x)
        measurer = tq.MeasureAll(obs=tq.PauliZ)
        # flip because qiskit is from N to 0, tq is from 0 to N
        measured_tq = np.flip(measurer(q_dev).data[0].numpy())

        # qiskit
        circ = tq2qiskit(q_layer, x)
        circ.measure(list(range(n_wires)), list(range(n_wires)))

        # Select the QasmSimulator from the Aer provider
        simulator = Aer.get_backend("qasm_simulator")

        # Execute and get counts
        result = execute(circ, simulator, shots=1000000).result()
        counts = result.get_counts(circ)
        measured_qiskit = get_expectations_from_counts(counts, n_wires=n_wires)

        try:
            # WARNING: the measurement has randomness, so tolerate larger
            # differences (MAX 20%) between tq and qiskit
            # typical mean difference is less than 1%
            diff = np.abs(measured_tq - measured_qiskit).mean()
            diff_ratio = (
                np.abs((measured_tq - measured_qiskit) / measured_qiskit)
            ).mean()
            logger.info(f"Diff: tq vs qiskit {diff} \t Diff Ratio: " f"{diff_ratio}")
            assert np.allclose(measured_tq, measured_qiskit, atol=1e-4, rtol=2e-1)
            logger.info(f"PASS tq vs qiskit [n_wires]={n_wires}")

        except AssertionError:
            logger.exception(f"FAIL tq vs qiskit [n_wires]={n_wires}")
            raise AssertionError

    logger.info(f"PASS tq vs qiskit measurement test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="pdb")
    args = parser.parse_args()

    # seed = 45
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    if args.pdb:
        pdb.set_trace()

    unitary_tq_vs_qiskit_test()
    state_tq_vs_qiskit_test()
    measurement_tq_vs_qiskit_test()
