import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim



class ODEFunc(nn.Module):

    def __init__(self,c_ops):
        super(ODEFunc,self).__init__()

        self.c_ops = c_ops

    def forward(self,t,y):
        return 

device = torch.device('cpu')

y0 = torch.tensor([1,0]).to(device)
t = torch.linspace(0., 1., 11).to(device)



c_ops = torch.linspace(0.,1.,11).to(device)

func = ODEFunc(c_ops).to(device)
res_y = odeint(func,y0,t).to(device)
