import torchquantum as tq
import qiskit
from qiskit_aer import Aer
from qiskit import transpile

from torchquantum.util import (
    switch_little_big_endian_matrix,
    find_global_phase,
)

from qiskit.circuit.library import GR, GRX, GRY, GRZ
import numpy as np

all_pairs = [
    {"qiskit": GR, "tq": tq.layer.GlobalR, "params": 2},
    {"qiskit": GRX, "tq": tq.layer.GlobalRX, "params": 1},
    {"qiskit": GRY, "tq": tq.layer.GlobalRY, "params": 1},
    {"qiskit": GRZ, "tq": tq.layer.GlobalRZ, "params": 1},
]

ITERATIONS = 2

def test_rotgates():
    # test each pair
    for pair in all_pairs:
        # test 2-5 wires
        for num_wires in range(2, 5):
            # try multiple random parameters
            for _ in range(ITERATIONS):
                # generate random parameters
                params = [
                    np.random.uniform(-2 * np.pi, 2 * np.pi) for _ in range(pair["params"])
                ]

                # create the qiskit circuit
                qiskit_circuit = pair["qiskit"](num_wires, *params)

                # get the unitary from qiskit
                backend = Aer.get_backend("unitary_simulator")
                qiskit_circuit = transpile(qiskit_circuit, backend)
                result = backend.run(qiskit_circuit).result()
                unitary_qiskit = result.get_unitary(qiskit_circuit)

                # create tq circuit
                qdev = tq.QuantumDevice(num_wires)
                tq_circuit = pair["tq"](num_wires, *params)
                tq_circuit(qdev)

                # get the unitary from tq
                unitary_tq = tq_circuit.get_unitary(qdev)
                unitary_tq = switch_little_big_endian_matrix(unitary_tq.data.numpy())

                # phase?
                phase = find_global_phase(unitary_tq, unitary_qiskit, 1e-4)

                assert np.allclose(
                    unitary_tq * phase, unitary_qiskit, atol=1e-6
                ), f"{pair} not equal with {params=}!"
