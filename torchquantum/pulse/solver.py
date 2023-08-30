import torch

class Solver(torch.nn.Module):
    def __init__(
        self,
        H,
        psi0,
        t_save,
        exp_ops,
        options
    ):
        self.H = H
        self.psi0 = psi0
        self.t_save = t_save
        self.exp_ops = exp_ops
        self.options = options


