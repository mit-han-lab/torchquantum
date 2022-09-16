import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

import torchquantum as tq
import torchquantum.functional as tqf
import pdb
import numpy as np

if __name__ == '__main__':
    pdb.set_trace()
    # target_unitary = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    theta = 0.6
    target_unitary = torch.tensor(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, np.cos(theta/2), -1j*np.sin(theta/2)],
         [0, 0, -1j*np.sin(theta/2), np.cos(theta/2)],
         ], dtype=torch.complex64)

    pulse_q0 = tq.QuantumPulseDirect(n_steps=10,
                                     hamil=[[0, 1], [1, 0]])
    pulse_q1 = tq.QuantumPulseDirect(n_steps=10,
                                     hamil=[[0, 1], [1, 0]])
    pulse_q01 = tq.QuantumPulseDirect(n_steps=10,
                                      hamil=[[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 0, 1],
                                             [0, 0, 1, 0],
                                             ]
                                      )

    optimizer = optim.Adam(params=list(pulse_q0.parameters()) + list(pulse_q1.parameters()) + list(pulse_q01.parameters()), lr=5e-3)

    for k in range(1000):
        u_0 = pulse_q0.get_unitary()
        u_1 = pulse_q1.get_unitary()
        u_01 = pulse_q01.get_unitary()
        # overall_u = u_01
        overall_u = torch.kron(u_0, u_1) @ u_01
        # loss = (abs(pulse.get_unitary() - target_unitary)**2).sum()
        loss = 1 - (torch.trace(overall_u @ target_unitary) / target_unitary.shape[0]).abs() ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(pulse.pulse_shape.grad)
        print(loss)
        # print(pulse.pulse_shape)
        # print(pulse.get_unitary())
        print(overall_u)
