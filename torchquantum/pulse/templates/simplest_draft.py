import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchquantum as tq

from torchdiffeq import odeint


# class SqaurePulse(Pulse):




# class Pulse(tq.QuantumModule):
#    def __init__():





# class Solver(ABC):
#    def __init__(
#       self,
#
#
#    ):


z_value = np.array([[1,0],[0,-1]])
z = torch.tensor(z_value,dtype=torch.complex64)

x_value = np.array([[0,1],[1,0]])
x = torch.tensor(x_value,dtype=torch.complex64)

y0_value = np.array([1,0])
y0 = torch.tensor(y0_value,dtype=torch.complex64)

dt = 0.22 #ns
# test 10 dt first: 2.2ns
# torch.nn.parameter.Parameter
pulse_value = torch.tensor([1,2,3,4,5,5,4,3,2,1]) / 10 #MHz
pulse = torch.nn.parameter.Parameter(pulse_value)
t_list = torch.tensor(list(range(10))) * dt

assert pulse.size() == t_list.size()

def H(t):
    t_ind = (t/dt).long()
    h = z + x * pulse[t_ind]
    # print("current ind:",t_ind)
    print("current h:",h)
    return h

def f(t, y):
    h = H(t)
    return -1.j * torch.matmul(h, y)
y_t = odeint(f, y0, t_list)
print("y0:",y0)
print("y_t:",y_t)
print("norm:",torch.norm(y_t,dim=1))

# calculate the gradients of pulse segments
# y_t[-1][0].real.backward()
# print(pulse.grad)
