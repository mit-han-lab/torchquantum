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

from qiskit import QuantumCircuit
import numpy as np
import random
from qiskit.quantum_info import Statevector, SparsePauliOp

import torchquantum as tq

from torchquantum.measurement import expval_joint_analytical, expval_joint_sampling
from torchquantum.plugin import op_history2qiskit
from torchquantum.util import switch_little_big_endian_state

import torch

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
        expval_tq_sampling = expval_joint_sampling(
            qdev, observable="".join(obs), n_shots=100000
        )[0].item()

        qiskit_circ = op_history2qiskit(qdev.n_wires, qdev.op_history)

        # opflow was little-endian, SparsePauliOp is also little-endian.
        # TorchQuantum is big-endian.
        # So we need to reverse the observable string for qiskit.
        qiskit_observable_str = "".join(reversed(obs))
        operator = SparsePauliOp(qiskit_observable_str)

        psi = Statevector.from_instruction(qiskit_circ)
        psi_evaled = psi.data
        state_tq = switch_little_big_endian_state(
            qdev.get_states_1d().detach().numpy()
        )[0]
        assert np.allclose(psi_evaled, state_tq, atol=1e-5)

        expval_qiskit = psi.expectation_value(operator).real
        # print(expval_tq, expval_qiskit)
        assert np.isclose(expval_tq, expval_qiskit, atol=1e-5)
        if (
            n_wires <= 3
        ):  # if too many wires, the stochastic method is not accurate due to limited shots
            assert np.isclose(expval_tq_sampling, expval_qiskit, atol=1e-2)

    print("expval observable test passed")


def util0():
    """from below we know that the Z ^ I means Z on qubit 1 and I on qubit 0"""
    qc = QuantumCircuit(2)

    qc.x(0)

    operator = SparsePauliOp("ZI")
    psi = Statevector(qc)
    expectation_value = psi.expectation_value(operator)
    print(expectation_value.real)
    # result: 1.0, means measurement result is 0, so Z is on qubit 1

    operator = SparsePauliOp("IZ")
    psi = Statevector(qc)
    expectation_value = psi.expectation_value(operator)
    print(expectation_value.real)
    # result: -1.0 means measurement result is 1, so Z is on qubit 0

    operator = SparsePauliOp("II")
    psi = Statevector(qc)
    expectation_value = psi.expectation_value(operator)
    print(expectation_value.real)

    operator = SparsePauliOp("ZZ")
    psi = Statevector(qc)
    expectation_value = psi.expectation_value(operator)
    print(expectation_value.real)

    qc = QuantumCircuit(3)

    qc.x(0)

    operator = SparsePauliOp("ZII")
    psi = Statevector(qc)
    expectation_value = psi.expectation_value(operator)
    print(expectation_value.real)


if __name__ == "__main__":
    import pdb

    pdb.set_trace()

    # util0()
    test_expval_observable()
