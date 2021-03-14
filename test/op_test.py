# test the torchquantum.functional against the IBM Qiskit
import argparse
import pdb
import torchquantum as tq
import numpy as np

from torchpack.utils.logging import logger
from torchquantum.utils import switch_little_big_endian_matrix
from tqdm import tqdm

import qiskit.circuit.library.standard_gates as qiskit_gate

RND_TIMES = 100

pair_list = [
    {'qiskit': qiskit_gate.HGate, 'tq': tq.Hadamard},
    {'qiskit': qiskit_gate.XGate, 'tq': tq.PauliX},
    {'qiskit': qiskit_gate.YGate, 'tq': tq.PauliY},
    {'qiskit': qiskit_gate.ZGate, 'tq': tq.PauliZ},
    {'qiskit': qiskit_gate.SGate, 'tq': tq.S},
    {'qiskit': qiskit_gate.TGate, 'tq': tq.T},
    {'qiskit': qiskit_gate.SXGate, 'tq': tq.SX},
    {'qiskit': qiskit_gate.CXGate, 'tq': tq.CNOT},
    {'qiskit': qiskit_gate.CYGate, 'tq': tq.CY},
    {'qiskit': qiskit_gate.CZGate, 'tq': tq.CZ},
    {'qiskit': qiskit_gate.RXGate, 'tq': tq.RX},
    {'qiskit': qiskit_gate.RYGate, 'tq': tq.RY},
    {'qiskit': qiskit_gate.RZGate, 'tq': tq.RZ},
    {'qiskit': qiskit_gate.SwapGate, 'tq': tq.SWAP},
    {'qiskit': qiskit_gate.CSwapGate, 'tq': tq.CSWAP},
    {'qiskit': qiskit_gate.CCXGate, 'tq': tq.Toffoli},
    {'qiskit': qiskit_gate.PhaseGate, 'tq': tq.PhaseShift},
    # {'qiskit': qiskit_gate.?, 'tq': tq.Rot},
    # {'qiskit': qiskit_gate.?, 'tq': tq.MultiRZ},
    {'qiskit': qiskit_gate.CRXGate, 'tq': tq.CRX},
    {'qiskit': qiskit_gate.CRYGate, 'tq': tq.CRY},
    {'qiskit': qiskit_gate.CRZGate, 'tq': tq.CRZ},
    # {'qiskit': qiskit_gate.?, 'tq': tq.CRot},
    {'qiskit': qiskit_gate.U1Gate, 'tq': tq.U1},
    {'qiskit': qiskit_gate.U2Gate, 'tq': tq.U2},
    {'qiskit': qiskit_gate.U3Gate, 'tq': tq.U3},
]


if __name__ == '__main__':
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', action='store_true', help='pdb')

    args = parser.parse_args()

    if args.pdb:
        pdb.set_trace()

    for pair in pair_list:
        try:
            if pair['tq'].num_params == 0:
                qiskit_matrix = pair['qiskit']().to_matrix()
                tq_matrix = pair['tq'].matrix.numpy()
                tq_matrix = switch_little_big_endian_matrix(tq_matrix)
                assert np.allclose(qiskit_matrix, tq_matrix)
            else:
                for k in tqdm(range(RND_TIMES)):
                    rnd_params = np.random.rand(pair['tq'].num_params).tolist()
                    qiskit_matrix = pair['qiskit'](*rnd_params).to_matrix()
                    tq_matrix = pair['tq'](
                        has_params=True,
                        trainable=False,
                        init_params=rnd_params).matrix.numpy()
                    tq_matrix = switch_little_big_endian_matrix(tq_matrix)
                    assert np.allclose(qiskit_matrix, tq_matrix)

            logger.info(f"Gate {pair['tq']().name} match.")
        except AssertionError:
            logger.exception(f"Gate {pair['tq']().name} not match.")
            raise AssertionError

