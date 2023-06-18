import torch
import math
from ..solver import Solver
from ..utils import *
from torchdiffeq import odeint

def mesolve(
    dens0,
    H=None,
    n_dt=None,
    dt=0.22,
    *,
    L_ops=None,
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

    args = (H, dens0, t_save, exp_ops, options)

    solver = MESolver(*args, L_ops=L_ops)

    solver.run()

    psi_save, exp_save = solver.y_save, solver.exp_save

    return psi_save, exp_save

def _lindblad_helper(L, rho):
    Ldag = torch.conj(L)
    return L @ rho @ Ldag - 0.5 * Ldag @ L @ rho - 0.5 * rho @ Ldag @ L

def lindbladian(H,rho,L_ops):
    if L_ops is None:
        return -1j * (H @ rho - rho @ H)

    if type(L_ops) is not list:
        L_ops = [L_ops]

    _dissipator = [_lindblad_helper(L, rho) for L in L_ops]
    dissipator = torch.stack(_dissipator)
    return -1j * (H @ rho - rho @ H) + dissipator.sum(0)

class MESolver(Solver):

    def __init__(self, *args, L_ops):
        super().__init__(*args)
        self.L_ops = L_ops


    def f(self, t, y):
        h = self.H(t)
        return lindbladian(h,y,self.L_ops)


    def run(self):
        # self.y_save = odeint(self.f, self.psi0, self.t_save, method='rk4', options=self.options)
        self.y_save = odeint(self.f, self.psi0, self.t_save)
        self.exp_save = None
