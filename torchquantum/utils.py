import numpy as np
import torch
import functools

def pauli_eigs(n):
    r"""Eigenvalues for :math:`A^{\o times n}`, where :math:`A` is
    Pauli operator, or shares its eigenvalues.

    As an example if n==2, then the eigenvalues of a tensor product consisting
    of two matrices sharing the eigenvalues with Pauli matrices is returned.

    Args:
        n (int): the number of qubits the matrix acts on
    Returns:
        list: the eigenvalues of the specified observable
    """
    if n == 1:
        return np.array([1, -1])
    return np.concatenate([pauli_eigs(n - 1), -pauli_eigs(n - 1)])


def diag(x):
    # input tensor, output tensor with diagonal as the input
    # manual implementation because torch.diag does not support autograd of
    # complex number
    diag_len = x.shape[-1]
    x = x.unsqueeze(-1)
    dims = list(x.shape)
    x = torch.cat([x, torch.zeros(dims[:-1] + [diag_len]).to(x)], dim=-1)
    x = x.view(dims[:-2] + [diag_len * (diag_len + 1)])[..., :-diag_len]
    x = x.view(dims[:-2]+[diag_len, diag_len])
    return x


class Timer(object):
    def __init__(self, device='gpu', name='', times=100):
        self.device = device
        self.name = name
        self.times = times
        if device == 'gpu':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.device == 'gpu':
            self.start.record()

    def __exit__(self):
        if self.device == 'gpu':
            self.end.record()
            torch.cuda.synchronize()
            print(f"Task: {self.name}: "
                  f"{self.start.elapsed_time(self.end) / self.times}")


def static_support(f):
    @functools.wraps(f)
    def forward_register_graph(*args, **kwargs):
        if args[0].static_mode and args[0].parent_graph is not None:
            args[0].parent_graph.add_op(args[0])
        res = f(*args, **kwargs)
        if args[0].static_mode and args[0].is_graph_top:
            # finish build graph, set flag
            args[0].set_graph_build_finish()
            args[0].static_forward(args[0].q_device)

        return res
    return forward_register_graph
