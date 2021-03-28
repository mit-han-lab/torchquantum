import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

from torchquantum.macro import C_DTYPE
from torchpack.utils.logging import logger
from typing import List, Dict


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
    exps = []
    if isinstance(counts, dict):
        counts = [counts]
    for count in counts:
        ctr_one = [0] * n_wires
        total_shots = 0
        for k, v in count.items():
            for wire in range(n_wires):
                if k[wire] == '1':
                    ctr_one[wire] += v
            total_shots += v
        prob_one = np.array(ctr_one) / total_shots
        exp = np.flip(-1 * prob_one + 1 * (1 - prob_one))
        exps.append(exp)
    res = np.stack(exps)
    return res


def find_global_phase(mat1, mat2, threshold):
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            # find a numerical stable global phase
            if np.abs(mat1[i][j]) > threshold and \
                    np.abs(mat1[i][j]) > threshold:
                return mat2[i][j] / mat1[i][j]
    return None


def build_module_description(m: tq.QuantumModule, x=None) -> dict:
    """
    serialize all operations in the module and generate a list with
    [{'name': RX, 'has_params': True, 'trainable': True, 'wires': [0],
    n_wires: 1, 'params': [array([[0.01]])]}]
    so that an identity module can be reconstructed
    The module needs to have static support
    """

    m.static_off()
    m.static_on(wires_per_block=None)
    m.is_graph_top = False

    # forward to register all modules and parameters
    if x is None:
        m.forward(q_device=None)
    else:
        m.forward(q_device=None, x=x)

    m.is_graph_top = True
    m.graph.build_flat_module_list()

    module_list = m.graph.flat_module_list
    m.static_off()

    desc = []

    for module in module_list:
        if module.params is not None:
            if module.params.shape[0] > 1:
                # more than one param, so it is from classical input with
                # batch mode
                assert not module.has_params
                params = None
            else:
                # has quantum params, batch has to be 1
                params = module.params[0].data.cpu().numpy()
        else:
            params = None

        desc.append({
            'name': module.name.lower(),
            'has_params': module.has_params,
            'trainable': module.trainable,
            'wires': module.wires,
            'n_wires': module.n_wires,
            'params': params
        })

    return desc


def build_module_from_description(desc: List[Dict]) -> tq.QuantumModule:
    ops = []
    for info in desc:
        op = tq.op_name_dict[info['name']](
            has_params=info['has_params'],
            trainable=info['trainable'],
            wires=info['wires'],
            n_wires=info['n_wires'],
            init_params=info['params'],
        )
        ops.append(op)

    return tq.QuantumModuleFromOps(ops)


def build_module_description_test():
    import pdb
    from torchquantum.plugins import tq2qiskit

    pdb.set_trace()
    from examples.core.models.q_models import QFCModel12
    q_model = QFCModel12({'n_blocks': 4})
    desc = build_module_description(q_model.q_layer)
    print(desc)
    q_dev = tq.QuantumDevice(n_wires=4)
    m = build_module_from_description(desc)
    tq2qiskit(q_dev, m, draw=True)

    desc = build_module_description(tq.RandomLayerAllTypes(
        n_ops=200,  wires=[0, 1, 2, 3], qiskit_compatible=True))
    print(desc)
    m1 = build_module_from_description(desc)
    tq2qiskit(q_dev, m1, draw=True)


if __name__ == '__main__':
    build_module_description_test()
    switch_little_big_endian_matrix_test()
    switch_little_big_endian_state_test()
