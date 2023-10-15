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

# test the torchquantum.functional against the IBM Qiskit
import argparse
import pdb
import torchquantum as tq
import numpy as np

from torchpack.utils.logging import logger
from torchquantum.util import switch_little_big_endian_matrix
from tqdm import tqdm

import qiskit.circuit.library.standard_gates as qiskit_gate
import qiskit.circuit.library as qiskit_library
from qiskit.quantum_info import Operator


RND_TIMES = 100

pair_list = [
    {"qiskit": qiskit_gate.HGate, "tq": tq.Hadamard},
    {"qiskit": None, "tq": tq.SHadamard},
    {"qiskit": qiskit_gate.XGate, "tq": tq.PauliX},
    {"qiskit": qiskit_gate.YGate, "tq": tq.PauliY},
    {"qiskit": qiskit_gate.ZGate, "tq": tq.PauliZ},
    {"qiskit": qiskit_gate.SGate, "tq": tq.S},
    {"qiskit": qiskit_gate.TGate, "tq": tq.T},
    {"qiskit": qiskit_gate.SXGate, "tq": tq.SX},
    {"qiskit": qiskit_gate.CXGate, "tq": tq.CNOT},
    {"qiskit": qiskit_gate.CYGate, "tq": tq.CY},
    {"qiskit": qiskit_gate.CZGate, "tq": tq.CZ},
    {"qiskit": qiskit_gate.RXGate, "tq": tq.RX},
    {"qiskit": qiskit_gate.RYGate, "tq": tq.RY},
    {"qiskit": qiskit_gate.RZGate, "tq": tq.RZ},
    {"qiskit": qiskit_gate.RXXGate, "tq": tq.RXX},
    {"qiskit": qiskit_gate.RYYGate, "tq": tq.RYY},
    {"qiskit": qiskit_gate.RZZGate, "tq": tq.RZZ},
    {"qiskit": qiskit_gate.RZXGate, "tq": tq.RZX},
    {"qiskit": qiskit_gate.SwapGate, "tq": tq.SWAP},
    # {'qiskit': qiskit_gate.?, 'tq': tq.SSWAP},
    {"qiskit": qiskit_gate.CSwapGate, "tq": tq.CSWAP},
    {"qiskit": qiskit_gate.CCXGate, "tq": tq.Toffoli},
    {"qiskit": qiskit_gate.PhaseGate, "tq": tq.PhaseShift},
    # {'qiskit': qiskit_gate.?, 'tq': tq.Rot},
    # {'qiskit': qiskit_gate.?, 'tq': tq.MultiRZ},
    {"qiskit": qiskit_gate.CRXGate, "tq": tq.CRX},
    {"qiskit": qiskit_gate.CRYGate, "tq": tq.CRY},
    {"qiskit": qiskit_gate.CRZGate, "tq": tq.CRZ},
    # {'qiskit': qiskit_gate.?, 'tq': tq.CRot},
    {"qiskit": qiskit_gate.UGate, "tq": tq.U},
    {"qiskit": qiskit_gate.U1Gate, "tq": tq.U1},
    {"qiskit": qiskit_gate.U2Gate, "tq": tq.U2},
    {"qiskit": qiskit_gate.U3Gate, "tq": tq.U3},
    {"qiskit": qiskit_gate.CUGate, "tq": tq.CU},
    {"qiskit": qiskit_gate.CU1Gate, "tq": tq.CU1},
    # {'qiskit': qiskit_gate.?, 'tq': tq.CU2},
    {"qiskit": qiskit_gate.CU3Gate, "tq": tq.CU3},
    {"qiskit": qiskit_gate.ECRGate, "tq": tq.ECR},
    # {"qiskit": qiskit_library.QFT, "tq": tq.QFT},
    {"qiskit": qiskit_gate.SdgGate, "tq": tq.SDG},
    {"qiskit": qiskit_gate.TdgGate, "tq": tq.TDG},
    {"qiskit": qiskit_gate.SXdgGate, "tq": tq.SXDG},
    {"qiskit": qiskit_gate.CHGate, "tq": tq.CH},
    {"qiskit": qiskit_gate.CCZGate, "tq": tq.CCZ},
    {"qiskit": qiskit_gate.iSwapGate, "tq": tq.ISWAP},
    {"qiskit": qiskit_gate.CSGate, "tq": tq.CS},
    {"qiskit": qiskit_gate.CSdgGate, "tq": tq.CSDG},
    {"qiskit": qiskit_gate.CSXGate, "tq": tq.CSX},
    {"qiskit": qiskit_gate.DCXGate, "tq": tq.DCX},
    {"qiskit": qiskit_gate.XXMinusYYGate, "tq": tq.XXMINYY},
    {"qiskit": qiskit_gate.XXPlusYYGate, "tq": tq.XXPLUSYY},
    {"qiskit": qiskit_gate.C3XGate, "tq": tq.C3X},
    {"qiskit": qiskit_gate.RGate, "tq": tq.R},
    {"qiskit": qiskit_gate.C4XGate, "tq": tq.C4X},
    {"qiskit": qiskit_gate.RCCXGate, "tq": tq.RCCX},
    {"qiskit": qiskit_gate.RC3XGate, "tq": tq.RC3X},
    {"qiskit": qiskit_gate.GlobalPhaseGate, "tq": tq.GlobalPhase},
    {"qiskit": qiskit_gate.C3SXGate, "tq": tq.C3SX},
]

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def test_op():
    """
    For CNOT, the Qiskit matrix is [[1, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 0]]
    the torchquantum matrix is:
                                   [[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0]]
    because the in Qiskit higher qubit indices are more significant,
    while in torchquantum, the higher qubit indices are less significant in
    conversion between torchquantum and qiskit, need to be aware of this
    different. similar to ALL other gates.
    """

    for pair in pair_list:
        try:
            if pair["tq"].num_params == 0:
                if pair["tq"]().name == "SHadamard":
                    """Square root of Hadamard is RY(pi/4)"""
                    qiskit_matrix = qiskit_gate.RYGate(theta=np.pi / 4).to_matrix()
                elif pair["tq"]().name == "C3SX":
                    qiskit_matrix = Operator(pair["qiskit"]())
                else:
                    qiskit_matrix = pair["qiskit"]().to_matrix()
                tq_matrix = pair["tq"].matrix.numpy()
                tq_matrix = switch_little_big_endian_matrix(tq_matrix)
                assert np.allclose(qiskit_matrix, tq_matrix)
            else:
                for k in tqdm(range(RND_TIMES)):
                    rnd_params = np.random.rand(pair["tq"].num_params).tolist()
                    qiskit_matrix = pair["qiskit"](*rnd_params).to_matrix()
                    tq_matrix = pair["tq"](
                        has_params=True, trainable=False, init_params=rnd_params
                    ).matrix.numpy()
                    tq_matrix = switch_little_big_endian_matrix(tq_matrix)
                    assert np.allclose(qiskit_matrix, tq_matrix)

            logger.info(f"Gate {pair['tq']().name} match.")
        except AssertionError:
            logger.exception(f"Gate {pair['tq']().name} not match.")
            raise AssertionError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="pdb")

    args = parser.parse_args()

    if args.pdb:
        pdb.set_trace()
    test_op()
