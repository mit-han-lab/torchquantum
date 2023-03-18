# import torchquantum as tq
#
# from torchquantum.plugins import op_history2qiskit
# from qiskit import Aer, transpile
# import numpy as np
#
# def test_measure():
#
#     n_shots = 10000
#     qdev = tq.QuantumDevice(n_wires=3, bsz=1, record_op=True)
#     qdev.x(wires=2) # type: ignore
#     qdev.x(wires=1) # type: ignore
#     qdev.ry(wires=0, params=0.98) # type: ignore
#     qdev.rx(wires=1, params=1.2) # type: ignore
#     qdev.cnot(wires=[0, 2]) # type: ignore
#
#     tq_counts = tq.measure(qdev, n_shots=n_shots)
#
#     circ = op_history2qiskit(qdev.n_wires, qdev.op_history)
#     circ.measure_all()
#     simulator = Aer.get_backend('aer_simulator')
#     circ = transpile(circ, simulator)
#     qiskit_res = simulator.run(circ, shots=n_shots).result()
#     qiskit_counts = qiskit_res.get_counts()
#
#     for k, v in tq_counts[0].items():
#         # need to reverse the bitstring because qiskit is in little endian
#         qiskit_ratio = qiskit_counts.get(k[::-1], 0) / n_shots
#         tq_ratio = v / n_shots
#         print(k, qiskit_ratio, tq_ratio)
#         assert np.isclose(qiskit_ratio, tq_ratio, atol=0.1)
#
#     print("tq.measure test passed")
#
# if __name__ == '__main__':
#     import pdb
#     pdb.set_trace()
#     test_measure()

import qiskit

from qiskit import IBMQ
import pdb

pdb.set_trace()

IBMQ.load_account()

provider = IBMQ.get_provider(hub="ibm-q")
backend = provider.get_backend("ibmq_belem")

# print(backend.defaults().instruction_schedule_map._map)


# https://qiskit.org/documentation/tutorials/circuits_advanced/08_gathering_system_information.html
def describe_qubit(qubit, properties):
    """Print a string describing some of reported properties of the given qubit."""

    # Conversion factors from standard SI units
    us = 1e6
    ns = 1e9
    GHz = 1e-9

    print(
        "Qubit {0} has a \n"
        "  - T1 time of {1} microseconds\n"
        "  - T2 time of {2} microseconds\n"
        "  - U2 gate error of {3}\n"
        "  - U2 gate duration of {4} nanoseconds\n"
        "  - resonant frequency of {5} GHz".format(
            qubit,
            properties.t1(qubit) * us,
            properties.t2(qubit) * us,
            properties.gate_error("sx", qubit),
            properties.gate_length("sx", qubit) * ns,
            properties.frequency(qubit) * GHz,
        )
    )


props = backend.properties()
describe_qubit(0, props)


def get_2q_errors(props):
    """Print the 2-qubit gate fidelities for the given backend."""
    errors = {}
    for gate in props.gates:
        if len(gate.qubits) == 2:
            errors[gate.name] = gate.parameters[0]
    return errors


print(get_2q_errors(props))
