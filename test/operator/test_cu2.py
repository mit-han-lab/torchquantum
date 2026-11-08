import numpy as np
import torch
import torchquantum as tq
from qiskit.circuit.library import U2Gate
from qiskit.quantum_info import Operator


def test_cu2_matrix():
    """CU2 must equal a controlled U2 (and in particular be unitary)."""
    phi, lam = 0.3, 0.7
    op = tq.CU2(
        has_params=True,
        trainable=False,
        init_params=torch.tensor([[phi, lam]]),
    )
    expected = Operator(U2Gate(phi, lam).control(1)).data
    # qiskit orders the control qubit as qubit 0 (least significant); tq
    # uses control as the first wire (most significant), so swap the two.
    swap = np.eye(4)[[0, 2, 1, 3]]
    expected = swap @ expected @ swap
    np.testing.assert_allclose(op.matrix.numpy(), expected, atol=1e-6)


if __name__ == "__main__":
    test_cu2_matrix()
