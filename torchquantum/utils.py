import functools
import numpy as np
import torch

def pauli_eigs(n):
    r"""Eigenvalues for :math:`A^{\otimes n}`, where :math:`A` is
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
