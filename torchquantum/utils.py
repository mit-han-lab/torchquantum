import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchquantum.macro import C_DTYPE


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
    x = torch.cat([x, torch.zeros(dims[:-1] + [diag_len]).to(x.device)],
                  dim=-1)
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

    def __exit__(self, exc_type, exc_value, tb):
        if self.device == 'gpu':
            self.end.record()
            torch.cuda.synchronize()
            print(f"Task: {self.name}: "
                  f"{self.start.elapsed_time(self.end) / self.times} ms")


def get_unitary_loss(model: nn.Module):
    loss = 0
    for name, params in model.named_parameters():
        if 'TrainableUnitary' in name:
            U = params
            like_identity = U.matmul(U.conj().permute(1, 0))
            identity = torch.eye(U.shape[0], dtype=C_DTYPE,
                                 device=U.device)
            loss += F.mse_loss(torch.view_as_real(identity),
                               torch.view_as_real(like_identity))

    return loss


def legalize_unitary(model: nn.Module):
    with torch.no_grad():
        for name, params in model.named_parameters():
            if 'TrainableUnitary' in name:
                U = params
                U, Sigma, V = torch.svd(U)
                params.data.copy_(U.matmul(V.conj().permute(1, 0)))
