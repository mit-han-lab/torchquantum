import torch
import math
from ..solver import Solver
from ..utils import *
from torchdiffeq import odeint

def sesolve(
    psi0,
    H=None,
    n_dt=None,
    dt=0.22,
    *,
    exp_ops=None,
    options=None,
    dtype=None,
    device=None
):
    if options is None:
        options = {}

    if not 'step_size' in options:
        options['step_size'] = 0.001

    t_save = torch.tensor(list(range(n_dt)))*dt

    args = (H, psi0, t_save, exp_ops, options)

    solver = SESolver(*args)

    solver.run()

    psi_save, exp_save = solver.y_save, solver.exp_save
    return psi_save, exp_save



class SESolver(Solver):

    def __init__(self, *args):
        super().__init__(*args)


    def f(self, t, y):
        h = self.H(t)
        return -1.j * torch.matmul(h, y)


    def run(self):
        # self.y_save = odeint(self.f, self.psi0, self.t_save, method='rk4', options=self.options)
        self.y_save = odeint(self.f, self.psi0, self.t_save)
        self.exp_save = None
