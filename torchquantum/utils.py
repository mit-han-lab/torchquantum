import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchquantum.macro import C_DTYPE
from torchpack.utils.logging import logger


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
    x = x.view(dims[:-2] + [diag_len, diag_len])
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
            like_identity = U.matmul(U.conj().permute(0, 2, 1))
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
                params.data.copy_(U.matmul(V.conj().permute(0, 2, 1)))


def switch_little_big_endian_matrix(mat):
    if len(mat.shape) % 2 == 1:
        is_batch_matrix = True
        bsz = mat.shape[0]
        reshape = [bsz] + [2] * int(np.log2(mat[0].size))
    else:
        is_batch_matrix = False
        reshape = [2] * int(np.log2(mat.size))

    original_shape = mat.shape
    mat = mat.reshape(reshape)
    axes = list(range(len(mat.shape) // 2))
    axes.reverse()
    axes += [axis + len(mat.shape) // 2 for axis in axes]

    if is_batch_matrix:
        axes = [0] + [axis + 1 for axis in axes]

    mat = np.transpose(mat, axes=axes).reshape(original_shape)
    return mat


def switch_little_big_endian_state(state):
    if len(state.shape) > 1:
        is_batch_state = True
        bsz = state.shape[0]
        reshape = [bsz] + [2] * int(np.log2(state[0].size))
    elif len(state.shape) == 1:
        is_batch_state = False
        reshape = [2] * int(np.log2(state.size))
    else:
        logger.exception(f"Dimension of statevector should be 1 or 2")
        raise ValueError

    original_shape = state.shape
    state = state.reshape(reshape)

    if is_batch_state:
        axes = list(range(1, len(state.shape)))
        axes.reverse()
        axes = [0] + axes
    else:
        axes = list(range(len(state.shape)))
        axes.reverse()

    mat = np.transpose(state, axes=axes).reshape(original_shape)

    return mat


def switch_little_big_endian_matrix_test():
    logger.info(switch_little_big_endian_matrix(np.ones((16, 16))))
    logger.info(switch_little_big_endian_matrix(np.ones((5, 16, 16))))


def switch_little_big_endian_state_test():
    logger.info(switch_little_big_endian_state(np.ones((5, 16))))
    logger.info(switch_little_big_endian_state(np.arange(8)))


def get_expectations_from_counts(counts, n_wires):
    ctr_one = [0] * n_wires
    total_shots = 0
    for k, v in counts.items():
        for wire in range(n_wires):
            if k[wire] == '1':
                ctr_one[wire] += v
        total_shots += v
    prob_one = np.array(ctr_one) / total_shots

    return -1 * prob_one + 1 * (1 - prob_one)


if __name__ == '__main__':
    switch_little_big_endian_matrix_test()
    switch_little_big_endian_state_test()
