from qiskit import QuantumCircuit
import numpy as np
import random
from qiskit.opflow import StateFn, X, Y, Z, I

import torchquantum as tq

from torchquantum.measurement import expval_joint_analytical
from torchquantum.plugins import op_history2qiskit
from torchquantum.utils import switch_little_big_endian_state

import torch

pauli_str_op_dict = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "I": I,
}


def test_expval_observable():
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    for k in range(100):
        # print(k)
        n_wires = random.randint(1, 10)
        obs = random.choices(["X", "Y", "Z", "I"], k=n_wires)
        random_layer = tq.RandomLayer(n_ops=100, wires=list(range(n_wires)))
        qdev = tq.QuantumDevice(n_wires=n_wires, bsz=1, record_op=True)
        random_layer(qdev)

        expval_tq = expval_joint_analytical(qdev, observable="".join(obs))[0].item()
        qiskit_circ = op_history2qiskit(qdev.n_wires, qdev.op_history)
        operator = pauli_str_op_dict[obs[0]]
        for ob in obs[1:]:
            # note here the order is reversed because qiskit is in little endian
            operator = pauli_str_op_dict[ob] ^ operator
        psi = StateFn(qiskit_circ)
        psi_evaled = psi.eval()._primitive._data
        state_tq = switch_little_big_endian_state(
            qdev.get_states_1d().detach().numpy()
        )[0]
        assert np.allclose(psi_evaled, state_tq, atol=1e-5)

        expval_qiskit = (~psi @ operator @ psi).eval().real
        # print(expval_tq, expval_qiskit)
        assert np.isclose(expval_tq, expval_qiskit, atol=1e-5)
    print("expval observable test passed")


def util0():
    """from below we know that the Z ^ I means Z on qubit 1 and I on qubit 0"""
    qc = QuantumCircuit(2)

    qc.x(0)

    operator = Z ^ I
    psi = StateFn(qc)
    expectation_value = (~psi @ operator @ psi).eval()
    print(expectation_value.real)
    # result: 1.0, means measurement result is 0, so Z is on qubit 1

    operator = I ^ Z
    psi = StateFn(qc)
    expectation_value = (~psi @ operator @ psi).eval()
    print(expectation_value.real)
    # result: -1.0 means measurement result is 1, so Z is on qubit 0

    operator = I ^ I
    psi = StateFn(qc)
    expectation_value = (~psi @ operator @ psi).eval()
    print(expectation_value.real)

    operator = Z ^ Z
    psi = StateFn(qc)
    expectation_value = (~psi @ operator @ psi).eval()
    print(expectation_value.real)

    qc = QuantumCircuit(3)

    qc.x(0)

    operator = I ^ I ^ Z
    psi = StateFn(qc)
    expectation_value = (~psi @ operator @ psi).eval()
    print(expectation_value.real)


if __name__ == "__main__":
    import pdb

    pdb.set_trace()

    # util0()
    test_expval_observable()
