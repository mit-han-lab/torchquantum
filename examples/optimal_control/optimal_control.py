import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

import torchquantum as tq
import torchquantum.functional as tqf
import pdb
import numpy as np

if __name__=='__main__':
    pdb.set_trace()
    # target_unitary = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    theta = 0.6
    target_unitary = torch.tensor([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=torch.complex64)

    pulse = tq.QuantumPulseDirect(n_steps=4,
                                  hamil=[[0, 1], [1, 0]])

    optimizer = optim.Adam(params=pulse.parameters(),  lr=5e-3)

    for k in range(1000):
        # loss = (abs(pulse.get_unitary() - target_unitary)**2).sum()
        loss = 1 - (torch.trace(pulse.get_unitary() @ target_unitary) / target_unitary.shape[0]).abs() ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(pulse.pulse_shape.grad)
        # print(loss)
        print(pulse.pulse_shape)
        print(pulse.get_unitary())
