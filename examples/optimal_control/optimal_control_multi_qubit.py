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

import torch
import torch.optim as optim

import torchquantum as tq
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="debug with pdb")
    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of training epochs"
    )

    args = parser.parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()

    # target_unitary = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    theta = 0.6
    target_unitary = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [0, 0, -1j * np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=torch.complex64,
    )

    pulse_q0 = tq.pulse.QuantumPulseDirect(n_steps=10, hamil=[[0, 1], [1, 0]])
    pulse_q1 = tq.pulse.QuantumPulseDirect(n_steps=10, hamil=[[0, 1], [1, 0]])
    pulse_q01 = tq.pulse.QuantumPulseDirect(
        n_steps=10,
        hamil=[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
    )

    optimizer = optim.Adam(
        params=list(pulse_q0.parameters())
        + list(pulse_q1.parameters())
        + list(pulse_q01.parameters()),
        lr=5e-3,
    )

    for k in range(args.epochs):
        u_0 = pulse_q0.get_unitary()
        u_1 = pulse_q1.get_unitary()
        u_01 = pulse_q01.get_unitary()
        # overall_u = u_01
        overall_u = torch.kron(u_0, u_1) @ u_01
        # loss = (abs(pulse.get_unitary() - target_unitary)**2).sum()
        loss = (
            1
            - (torch.trace(overall_u @ target_unitary) / target_unitary.shape[0]).abs()
            ** 2
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(pulse.pulse_shape.grad)
        print(loss)
        # print(pulse.pulse_shape)
        # print(pulse.get_unitary())
        print(overall_u)
