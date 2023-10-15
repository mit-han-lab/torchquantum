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

from torchquantum.operator import OpHamilExp
from torchquantum.algorithm import Hamiltonian
import numpy as np
from test.utils import check_all_close
from torchquantum.device import QuantumDevice


def test_op_hamil_exp():
    hamil = Hamiltonian(coeffs=[1.0, 0.5], paulis=["ZZ", "XX"])
    op = OpHamilExp(hamil=hamil, trainable=True, theta=0.45)

    print(op.matrix)
    print(op.exponent_matrix)

    check_all_close(
        op.matrix,
        np.array(
            [
                [
                    0.9686 - 0.2217j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    -0.0250 - 0.1094j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.9686 + 0.2217j,
                    0.0250 - 0.1094j,
                    0.0000 + 0.0000j,
                ],
                [
                    0.0000 + 0.0000j,
                    0.0250 - 0.1094j,
                    0.9686 + 0.2217j,
                    0.0000 + 0.0000j,
                ],
                [
                    -0.0250 - 0.1094j,
                    0.0000 + 0.0000j,
                    0.0000 + 0.0000j,
                    0.9686 - 0.2217j,
                ],
            ]
        ),
    )

    check_all_close(
        op.exponent_matrix,
        np.array(
            [
                [0.0 - 0.2250j, 0.0 + 0.0000j, 0.0 + 0.0000j, 0.0 - 0.1125j],
                [0.0 + 0.0000j, 0.0 + 0.2250j, 0.0 - 0.1125j, 0.0 + 0.0000j],
                [0.0 + 0.0000j, 0.0 - 0.1125j, 0.0 + 0.2250j, 0.0 + 0.0000j],
                [0.0 - 0.1125j, 0.0 + 0.0000j, 0.0 + 0.0000j, 0.0 - 0.2250j],
            ]
        ),
    )

    qdev = QuantumDevice(n_wires=2)
    qdev.reset_states(bsz=2)

    op(qdev, wires=[1, 0])

    print(qdev.get_states_1d().cpu().detach().numpy())

    check_all_close(
        qdev.get_states_1d().cpu().detach().numpy(),
        np.array(
            [
                [
                    0.9686322 - 0.22169423j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -0.02504631 - 0.1094314j,
                ],
                [
                    0.9686322 - 0.22169423j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -0.02504631 - 0.1094314j,
                ],
            ]
        ),
    )


if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()
    test_op_hamil_exp()
