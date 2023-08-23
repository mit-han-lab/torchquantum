import numpy as np
import torch
import torchquantum as tq
from torchquantum.pulse import sigmax, sigmay, sigmaz, sesolve


n_dt = 10
dt = 0.22


initial_value = np.array([1,0])
initial_state = torch.tensor(initial_value,dtype=torch.complex64)


pulse_value = torch.tensor(np.ones((n_dt,1)),dtype=torch.complex64)
pulse = torch.nn.parameter.Parameter(pulse_value)


def H(t):
    t_ind = (t/dt).long()
    return -sigmaz() + sigmax() * pulse[t_ind]


result = sesolve(psi0 = initial_state, H = H, n_dt = n_dt, dt = dt)


print(result)
