import copy
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from qiskit import IBMQ
from qiskit.exceptions import QiskitError
from qiskit.providers.aer.noise.device.parameters import gate_error_values
from torchpack.utils.config import Config
from torchpack.utils.logging import logger

import torchquantum as tq
from torchquantum.macro import C_DTYPE


def pauli_eigs(n) -> np.ndarray:
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
                    np.abs(mat2[i][j]) > threshold:
                return mat2[i][j] / mat1[i][j]
    return None


def build_module_op_list(m: tq.QuantumModule, x=None) -> List:
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

    op_list = []

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

        op_list.append({
            'name': module.name.lower(),
            'has_params': module.has_params,
            'trainable': module.trainable,
            'wires': module.wires,
            'n_wires': module.n_wires,
            'params': params
        })

    return op_list


def build_module_from_op_list(op_list: List[Dict],
                              remove_ops=False,
                              thres=None) -> tq.QuantumModule:
    logger.info(f"Building module from op_list...")
    thres = 1e-5 if thres is None else thres
    n_removed_ops = 0
    ops = []
    for info in op_list:
        params = info['params']

        if remove_ops:
            if params is not None:
                params = np.array(params) if isinstance(params, Iterable) \
                    else np.array([params])
                params = params % (2 * np.pi)
                params[params > np.pi] -= 2 * np.pi
                if all(abs(params) < thres):
                    n_removed_ops += 1
                    continue

        op = tq.op_name_dict[info['name']](
            has_params=info['has_params'],
            trainable=info['trainable'],
            wires=info['wires'],
            n_wires=info['n_wires'],
            init_params=info['params'],
        )
        ops.append(op)

    if n_removed_ops > 0:
        logger.warning(f"Remove in total {n_removed_ops} pruned operations.")
    else:
        logger.info(f"Do not remove any operations.")

    return tq.QuantumModuleFromOps(ops)


def build_module_description_test():
    import pdb

    from torchquantum.plugins import tq2qiskit

    pdb.set_trace()
    from examples.core.models.q_models import QFCModel12
    q_model = QFCModel12({'n_blocks': 4})
    desc = build_module_op_list(q_model.q_layer)
    print(desc)
    q_dev = tq.QuantumDevice(n_wires=4)
    m = build_module_from_op_list(desc)
    tq2qiskit(q_dev, m, draw=True)

    desc = build_module_op_list(tq.RandomLayerAllTypes(
        n_ops=200,  wires=[0, 1, 2, 3], qiskit_compatible=True))
    print(desc)
    m1 = build_module_from_op_list(desc)
    tq2qiskit(q_dev, m1, draw=True)


def get_p_v_reg_mapping(circ):
    """
    p are physical qubits
    v are logical qubits
    """
    p2v_orig = circ._layout.get_physical_bits().copy()
    mapping = {
        'p2v': {},
        'v2p': {},
    }

    for p, v in p2v_orig.items():
        if v.register.name == 'q':
            mapping['p2v'][p] = v.index
            mapping['v2p'][v.index] = p

    return mapping


def get_p_c_reg_mapping(circ):
    """
    p are physical qubits
    c are classical registers
    """
    mapping = {
        'p2c': {},
        'c2p': {},
    }
    for gate in circ.data:
        if gate[0].name == 'measure':
            mapping['p2c'][gate[1][0].index] = gate[2][0].index
            mapping['c2p'][gate[2][0].index] = gate[1][0].index

    return mapping


def get_v_c_reg_mapping(circ):
    """
    p are physical qubits, the real fabricated qubits
    v are logical qubits, also the 'wires' in torchquantum lib
    c are classical registers
    want to get v2c
    """

    p2v_orig = circ._layout.get_physical_bits().copy()
    p2v = {}
    for p, v in p2v_orig.items():
        if v.register.name == 'q':
            p2v[p] = v.index

    mapping = {
        'p2c': {},
        'c2p': {},
    }
    for gate in circ.data:
        if gate[0].name == 'measure':
            mapping['p2c'][gate[1][0].index] = gate[2][0].index
            mapping['c2p'][gate[2][0].index] = gate[1][0].index

    mapping2 = {
        'v2c': {},
        'c2v': {}
    }

    for c, p in mapping['c2p'].items():
        mapping2['c2v'][c] = p2v[p]

    for c, v in mapping2['c2v'].items():
        mapping2['v2c'][v] = c

    return mapping2


def get_cared_configs(conf, mode) -> Config:
    """only preserve cared configs"""
    conf = copy.deepcopy(conf)
    ignores = ['callbacks',
               'criterion',
               'debug',
               'legalization',
               'regularization',
               'verbose',
               'get_n_params',
               'prune',
               ]

    if 'super' not in conf.trainer.name:
        ignores.append('scheduler')
        ignores.append('optimizer')

    for ignore in ignores:
        if hasattr(conf, ignore):
            delattr(conf, ignore)

    if hasattr(conf, 'dataset'):
        dataset_ignores = ['binarize',
                           'binarize_threshold',
                           'center_crop',
                           'name',
                           'resize',
                           'resize_mode',
                           'root',
                           'train_valid_split_ratio',
                           ]
        for dataset_ignore in dataset_ignores:
            if hasattr(conf.dataset, dataset_ignore):
                delattr(conf.dataset, dataset_ignore)

    if not mode == 'es' and hasattr(conf, 'es'):
        delattr(conf, 'es')
    elif mode == 'es' and hasattr(conf, 'es') and hasattr(conf.es, 'eval'):
        delattr(conf.es, 'eval')

    if not mode == 'train' and hasattr(conf, 'trainer'):
        delattr(conf, 'trainer')

    if hasattr(conf, 'qiskit'):
        qiskit_ignores = [
                          'seed_simulator',
                          'seed_transpiler',
                          'coupling_map_name',
                          'basis_gates_name',
                          'est_success_rate',
                          ]
        for qiskit_ignore in qiskit_ignores:
            if hasattr(conf.qiskit, qiskit_ignore):
                delattr(conf.qiskit, qiskit_ignore)

    if hasattr(conf, 'run'):
        run_ignores = ['device',
                       'workers_per_gpu',
                       'n_epochs']
        for run_ignore in run_ignores:
            if hasattr(conf.run, run_ignore):
                delattr(conf.run, run_ignore)

    return conf


def get_success_rate(properties, transpiled_circ):
    # estimate the success rate according to the error rates of single and
    # two-qubit gates in transpiled circuits

    gate_errors = gate_error_values(properties)
    # construct the error dict
    gate_error_dict = {}
    for gate_error in gate_errors:
        if gate_error[0] not in gate_error_dict.keys():
            gate_error_dict[gate_error[0]] = {tuple(gate_error[1]):
                                              gate_error[2]}
        else:
            gate_error_dict[gate_error[0]][tuple(gate_error[1])] = \
                gate_error[2]

    success_rate = 1
    for gate in transpiled_circ.data:
        gate_success_rate = 1 - gate_error_dict[gate[0].name][tuple(
            map(lambda x: x.index, gate[1])
        )]
        if gate_success_rate == 0:
            gate_success_rate = 1e-5
        success_rate *= gate_success_rate

    return success_rate



def get_provider(backend_name, hub=None):
    # mass-inst-tech-1 or MIT-1
    if backend_name in ['ibmq_casablanca',
                        'ibmq_rome',
                        'ibmq_bogota',
                        'ibmq_jakarta']:
        if hub == 'mass' or hub is None:
            provider = IBMQ.get_provider(hub='ibm-q-research',
                                         group='mass-inst-tech-1',
                                         project='main')
        elif hub == 'mit':
            provider = IBMQ.get_provider(hub='ibm-q-research',
                                         group='MIT-1',
                                         project='main')
        else:
            raise ValueError(f"not supported backend {backend_name} in hub "
                             f"{hub}")
    elif backend_name in ['ibmq_paris',
                          'ibmq_toronto',
                          'ibmq_manhattan',
                          'ibmq_guadalupe',
                          'ibmq_montreal',
                          ]:
        provider = IBMQ.get_provider(hub='ibm-q-ornl',
                                     group='anl',
                                     project='csc428')
    else:
        if hub == 'mass' or hub is None:
            try:
                provider = IBMQ.get_provider(hub='ibm-q-research',
                                             group='mass-inst-tech-1',
                                             project='main')
            except QiskitError:
                # logger.warning(f"Cannot use MIT backend, roll back to open")
                logger.warning(f"Use the open backend")
                provider = IBMQ.get_provider(hub='ibm-q')
        elif hub == 'mit':
            provider = IBMQ.get_provider(hub='ibm-q-research',
                                         group='MIT-1',
                                         project='main')
        else:
            provider = IBMQ.get_provider(hub='ibm-q')

    return provider


def get_provider_hub_group_project(hub='ibm-q',
                                   group='open',
                                   project='main'):
    provider = IBMQ.get_provider(
        hub=hub,
        group=group,
        project=project,
    )
    return provider


def normalize_statevector(states):
    # make sure the square magnitude of statevector sum to 1
    # states = states.contiguous()
    original_shape = states.shape
    states_reshape = states.reshape(states.shape[0], -1)

    # for states with no energy, need to set all zero state as energy 1
    energy = (abs(states_reshape) ** 2).sum(dim=-1)
    if energy.min() == 0:
        for k, val in enumerate(energy):
            if val == 0:
                states_reshape[k][0] = 1

    factors = torch.sqrt(1 / ((abs(states_reshape) ** 2).sum(
        dim=-1))).unsqueeze(-1)
    states = (states_reshape * factors).reshape(original_shape)

    return states


def get_circ_stats(circ):
    depth = circ.depth()
    width = circ.width()
    size = circ.size()
    n_single_gates = 0
    n_two_gates = 0
    n_three_more_gates = 0
    n_gates_dict = {}
    n_measure = 0

    for gate in circ.data:
        op_name = gate[0].name
        wires = list(map(lambda x: x.index, gate[1]))
        if op_name in n_gates_dict.keys():
            n_gates_dict[op_name] += 1
        else:
            n_gates_dict[op_name] = 1

        if op_name == 'measure':
            n_measure += 1
        elif len(wires) == 1:
            n_single_gates += 1
        elif len(wires) == 2:
            n_two_gates += 1
        else:
            n_three_more_gates += 1

    return {
        'depth': depth,
        'size': size,
        'width': width,
        'n_single_gates': n_single_gates,
        'n_two_gates': n_two_gates,
        'n_three_more_gates': n_three_more_gates,
        'n_gates_dict': n_gates_dict
    }


def partial_trace(
    q_device: tq.QuantumDevice,
    keep_indices: List[int],
) -> torch.Tensor:
    """Returns a density matrix with only some qubits kept.
    Args:
        q_device: The q_device to take the partial trace over.
        keep_indices: Which indices to take the partial trace of the
            state_vector on.
    Returns:
        A density matrix with only the qubits specified by keep_indices.
    """
    n_wires = q_device.n_wires
    keep_indices = np.array(keep_indices) + 1
    dm_left_index = np.arange(n_wires + 1)
    dm_right_index = np.arange(n_wires + 1)
    dm_right_index[keep_indices] *= -1

    dm_out_index = np.array([0])
    dm_out_index = np.concatenate((dm_out_index, keep_indices))
    dm_out_index = np.concatenate((dm_out_index, -keep_indices))

    dm = contract(
        q_device.states,
        dm_left_index,
        q_device.states.conj(),
        dm_right_index,
        dm_out_index,
    )
    return dm


def tensor_form(dm: torch.Tensor):
    """Returns the tensor form of a density matrix.
    Args:
        dm: a (batched) density matrix.
    Returns:
        The tensor form of the density matrix.
    """
    size = dm.size()
    batched = len(size) % 2 == 1
    if batched:
        n_wires = int(np.log2(np.prod(size) / size[0]) / 2)
        shape_list = [size[0]] + [2] * (2 * n_wires)
    else:
        n_wires = int(np.log2(np.prod(size)) / 2)
        shape_list = [2] * (2 * n_wires)
    return dm.reshape(shape_list)


def matrix_form(dm: torch.Tensor):
    """Returns the matrix form of a density matrix.
    Args:
        dm: a (batched) density matrix.
    Returns:
        The matrix form of the density matrix.
    """
    size = dm.size()
    batched = len(size) % 2 == 1
    if batched:
        n_wires = int(np.log2(np.prod(size) / size[0]) / 2)
        shape_list = [size[0]] + [2**n_wires] * 2
    else:
        n_wires = int(np.log2(np.prod(size)) / 2)
        shape_list = [2**n_wires] * 2
    return dm.reshape(shape_list)


def dm_to_mixture_of_state(dm: torch.Tensor, atol=1e-10):
    """
    Args:
        q_device: The state vector to take the partial trace over.
        keep_indices: Which indices to take the partial trace of the
            state_vector on.
        atol: The tolerance for determining that a factored state is pure.
    """
    size = dm.size()
    batched = len(size) % 2 == 1
    if batched:
        dims = int(np.log2(np.prod(size) / size[0]) / 2)

        eigvals, eigvecs = torch.linalg.eigh(dm)
        result_list = []
        for batch in range(size[0]):
            mixture = tuple(
                zip(
                    eigvals[batch],
                    [vec.reshape([2] * dims) for vec in eigvecs[batch].T],
                )
            )
            result_list.append(
                tuple([(float(p[0]), p[1]) for p in mixture if p[0] > 1e-10])
            )
        return result_list
    else:

        dims = int(np.log2(np.prod(size)) / 2)

        eigvals, eigvecs = torch.linalg.eigh(dm)
        mixture = tuple(zip(eigvals, [vec.reshape([2] * dims) for vec in eigvecs.T]))
        return tuple([(float(p[0]), p[1]) for p in mixture if p[0] > 1e-10])

def partial_trace_test():
    import torchquantum.functional as tqf

    n_wires = 4
    q_device = tq.QuantumDevice(n_wires=n_wires)

    tqf.hadamard(q_device, wires=0)
    tqf.cnot(q_device, wires=[0, 1])

    dms = partial_trace(q_device, [0])
    dms = matrix_form(dms)
    mixture = dm_to_mixture_of_state(dms)
    
    print(mixture)
if __name__ == '__main__':
    build_module_description_test()
    switch_little_big_endian_matrix_test()
    switch_little_big_endian_state_test()
