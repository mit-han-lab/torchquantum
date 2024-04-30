import torchquantum as tq
from torchquantum.plugin import op_history2qiskit, qiskit2tq_op_history
from torchquantum.measurement import expval_joint_analytical
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Pauli
import numpy as np

"""
Testing strategy:
    partition on Operation: iterate through all the possible operations
    partition on number of gates in module: 1, >1
"""

def compare(ops, n_wires):
    # construct a normal tq circuit
    qmod = tq.QuantumModule.from_op_history(ops)
    qdev = tq.QuantumDevice(n_wires=n_wires, record_op=True)
    qmod(qdev)

    # turn into qiskit and inverse
    qiskit_circuit = op_history2qiskit(n_wires, qdev.op_history)
    qiskit_circuit = qiskit_circuit.inverse()

    # inverse the tq circuit
    qmod = tq.QuantumModule.from_op_history(ops)
    qdev = tq.QuantumDevice(n_wires=n_wires, record_op=True)
    qmod.inverse_module()
    qmod(qdev)

    qdev_ops = qiskit2tq_op_history(qiskit_circuit)

    for tq_op, qiskit_op in zip(qdev.op_history, qdev_ops):
        # TODO: name-wise (but currently need to ensure, e.g., cx == cnot)
        if tq_op["params"] is not None and qiskit_op["params"] is not None:
            assert np.allclose(np.array(tq_op["params"]), np.array(qiskit_op["params"]))

def get_random_rotations(num_params):
    return 4*np.pi*np.random.rand(num_params) - 2*np.pi

def test_inverse():
    ops = [
        {'name': 'u3', 'wires': 0, 'trainable': True, 'params': get_random_rotations(3)},
        {'name': 'u3', 'wires': 1, 'trainable': True, 'params': get_random_rotations(3)},
        {'name': 'cx', 'wires': [0, 1]},
        {'name': 'cx', 'wires': [1, 0]},
        {'name': 'u3', 'wires': 0, 'trainable': True, 'params': get_random_rotations(3)},
        {'name': 'u3', 'wires': 1, 'trainable': True, 'params': get_random_rotations(3)},
        {'name': 'cx', 'wires': [0, 1]},
        {'name': 'cx', 'wires': [1, 0]},
    ]
    compare(ops, 2)

# test_inverse()
