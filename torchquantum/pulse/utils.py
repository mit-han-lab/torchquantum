import numpy as np
import torch

def sigmax():
    x_value =  np.array([[0,1],[1,0]])
    return torch.tensor(x_value,dtype=torch.complex64)


def sigmay():
    y_value = np.array([[0,-1.j],[1.j,0]])
    return torch.tensor(y_value,dtype=torch.complex64)

def sigmaz():
    z_value = np.array([[1,0],[0,-1]])
    return torch.tensor(z_value,dtype=torch.complex64)


