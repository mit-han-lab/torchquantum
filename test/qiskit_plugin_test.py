import argparse
import pdb
import torch
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit.circuit.library.standard_gates as qiskit_gate
import numpy as np

from qiskit import QuantumCircuit
from qiskit import Aer, execute
from torchpack.utils.logging import logger
from torchquantum.utils import switch_little_big_endian_matrix
from test.static_mode_test import QLayer as AllRandomLayer
from torchquantum.plugins import tq2qiskit
from torchquantum.macro import F_DTYPE


def find_factor(unitary1, unitary2, threshold):
    for i in range(unitary1.shape[0]):
        for j in range(unitary1.shape[1]):
            # find a numerical stable factor
            if np.abs(unitary1[i][j]) > threshold and \
                    np.abs(unitary1[i][j]) > threshold:
                return unitary1[i][j] / unitary2[i][j]
    return None


def unitary_tq_vs_qiskit_test():
    n_wires = 2
    q_dev = tq.QuantumDevice(n_wires=n_wires)
    x = torch.randn((1, 100000), dtype=F_DTYPE)
    # q_layer = AllRandomLayer(n_wires=n_wires,
    #                          wires=list(range(n_wires)),
    #                          n_ops_rd=500,
    #                          n_ops_cin=0,
    #                          n_funcs=0,
    #                          qiskit_compatible=True)

    # q_layer = tq.RandomLayer(n_ops=500,
    #                          wires=list(range(n_wires)),
    #                          )

    q_layer = tq.RandomLayerAllTypes(n_ops=8,
                                     wires=list(range(n_wires)),
                                     qiskit_comparible=True,
                                     op_types=(
                                               # tq.Hadamard,
                                               # tq.PauliX,
                                               # tq.PauliY,
                                               # tq.PauliZ,
                                               # tq.S,
                                               # tq.T,
                                               # tq.SX,
                                               # tq.CNOT,
                                               # tq.CZ,
                                               # tq.CY,
                                               # tq.RX,
                                               # tq.RY,
                                               # tq.RZ,
                                               # tq.SWAP,
                                               # tq.CSWAP,
                                               # tq.Toffoli,
                                               # tq.PhaseShift,
                                               # tq.Rot,
                                               # tq.MultiRZ,
                                               tq.CRX,
                                               tq.CRY,
                                               # tq.CRZ,
                                               # tq.CRot,
                                               # tq.U1,
                                               # tq.U2,
                                               # tq.U3,
                                               # tq.MultiCNOT,
                                               # tq.MultiXCNOT,
                                               ),
                                     )
    # print(q_layer)
    # tq
    unitary_tq = q_layer.get_unitary(q_dev)
    unitary_tq = switch_little_big_endian_matrix(unitary_tq.data.numpy())

    # qiskit
    circ = tq2qiskit(q_layer)
    simulator = Aer.get_backend('unitary_simulator')
    result = execute(circ, simulator).result()
    unitary_qiskit = result.get_unitary(circ)

    stable_threshold = 1e-5
    try:
        # WARNING: need to remove the global phase! The qiskit simulated
        # results sometimes has global phase shift.
        factor = find_factor(unitary_tq, unitary_qiskit, stable_threshold)

        if factor is None:
            logger.exception(f"Cannot find a stable enough factor to reduce "
                             f"the global phase, increase the "
                             f"stable_threshold and try again")
            raise RuntimeError

        assert np.allclose(unitary_tq / factor, unitary_qiskit, atol=1e-6)
        logger.info(f"PASS tq vs qiskit [n_wires]={n_wires}")

    except AssertionError:
        logger.exception(f"FAIL tq vs qiskit [n_wires]={n_wires}")
        raise AssertionError

    except RuntimeError:
        raise RuntimeError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', action='store_true', help='pdb')
    args = parser.parse_args()

    seed = 45
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.pdb:
        pdb.set_trace()

    unitary_tq_vs_qiskit_test()
