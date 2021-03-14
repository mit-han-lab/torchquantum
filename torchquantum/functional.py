import functools
import torch
import torchquantum as tq
import numpy as np

from functools import partial
from typing import Callable
from .macro import C_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from .utils import pauli_eigs, diag
from torchpack.utils.logging import logger


__all__ = [
    'func_name_dict',
    'mat_dict',
    'apply_unitary_einsum',
    'apply_unitary_bmm',
    'hadamard',
    'paulix',
    'pauliy',
    'pauliz',
    's',
    't',
    'sx',
    'cnot',
    'cz',
    'cy',
    'rx',
    'ry',
    'rz',
    'swap',
    'cswap',
    'toffoli',
    'phaseshift',
    'rot',
    'multirz',
    'crx',
    'cry',
    'crz',
    'crot',
    'u1',
    'u2',
    'u3',
    'qubitunitary',
    'qubitunitaryfast',
    'qubitunitarystrict',
    'multicnot',
    'multixcnot',
    'x',
    'y',
    'z',
    'cx',
    'ccnot',
    'ccx',
]


def apply_unitary_einsum(state, mat, wires):
    device_wires = wires

    # minus one because of batch
    total_wires = len(state.shape) - 1

    if len(mat.shape) > 2:
        is_batch_unitary = True
        bsz = mat.shape[0]
        shape_extension = [bsz]
        # try:
        #     assert state.shape[0] == bsz
        # except AssertionError as err:
        #     logger.exception(f"Batch size of Quantum Device must be the same"
        #                      f" with that of gate unitary matrix")
        #     raise err

    else:
        is_batch_unitary = False
        shape_extension = []

    mat = mat.view(shape_extension + [2] * len(device_wires) * 2)

    mat = mat.type(C_DTYPE).to(state.device)

    # Tensor indices of the quantum state
    state_indices = ABC[: total_wires]

    # Indices of the quantum state affected by this operation
    affected_indices = "".join(ABC_ARRAY[list(device_wires)].tolist())

    # All affected indices will be summed over, so we need the same number
    # of new indices
    new_indices = ABC[total_wires: total_wires + len(device_wires)]

    # The new indices of the state are given by the old ones with the
    # affected indices replaced by the new_indices
    new_state_indices = functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0],
                                                        idx_pair[1]),
        zip(affected_indices, new_indices),
        state_indices,
    )

    # try:
    #     cannot support too many qubits...
    #     assert ABC[-1] not in state_indices + new_state_indices  \
    #      + new_indices + affected_indices
    # except AssertionError as err:
    #     logger.exception(f"Cannot support too many qubit.")
    #     raise err

    state_indices = ABC[-1] + state_indices
    new_state_indices = ABC[-1] + new_state_indices
    if is_batch_unitary:
        new_indices = ABC[-1] + new_indices

    # We now put together the indices in the notation numpy einsum
    # requires
    einsum_indices = f"{new_indices}{affected_indices}," \
                     f"{state_indices}->{new_state_indices}"

    new_state = torch.einsum(einsum_indices, mat, state)

    return new_state


def apply_unitary_bmm(state, mat, wires):
    device_wires = wires

    # if len(mat.shape) > 2:
    #         bsz = mat.shape[0]
    #     try:
    #         assert state.shape[0] == bsz
    #     except AssertionError as err:
    #         logger.exception(f"Batch size of Quantum Device must be the same"
    #                          f" with that of gate unitary matrix")
    #         raise err
    mat = mat.type(C_DTYPE).to(state.device)

    devices_dims = [w + 1 for w in device_wires]
    permute_to = list(range(state.dim()))
    for d in sorted(devices_dims, reverse=True):
        del permute_to[d]
    permute_to = permute_to[:1] + devices_dims + permute_to[1:]
    permute_back = list(np.argsort(permute_to))
    original_shape = state.shape
    permuted = state.permute(permute_to).reshape(
        [original_shape[0], mat.shape[-1], -1])

    if len(mat.shape) > 2:
        # both matrix and state are in batch mode
        new_state = mat.bmm(permuted)
    else:
        # matrix no batch, state in batch mode
        bsz = permuted.shape[0]
        expand_shape = [bsz] + list(mat.shape)
        new_state = mat.expand(expand_shape).bmm(permuted)

    new_state = new_state.view(original_shape).permute(permute_back)

    return new_state


def gate_wrapper(name, mat, method, q_device: tq.QuantumDevice, wires,
                 params=None, n_wires=None, static=False, parent_graph=None,
                 inverse=False):
    if params is not None:
        if not isinstance(params, torch.Tensor):
            # this is for qubitunitary gate
            params = torch.tensor(params, dtype=C_DTYPE)

        if name in ['qubitunitary', 'qubitunitaryfast', 'qubitunitarystrict']:
            params = params.unsqueeze(0) if params.dim() == 2 else params
        else:
            params = params.unsqueeze(-1) if params.dim() == 1 else params
    wires = [wires] if isinstance(wires, int) else wires

    if static:
        # in static mode, the function is not computed immediately, instead,
        # the unitary of a module will be computed and then applied
        parent_graph.add_func(name=name,
                              wires=wires,
                              parent_graph=parent_graph,
                              params=params,
                              n_wires=n_wires,
                              inverse=inverse)
    else:
        # in dynamic mode, the function is computed instantly
        if isinstance(mat, Callable):
            if n_wires is None or \
                    name in ['qubitunitary', 'qubitunitaryfast',
                             'qubitunitarystrict']:
                matrix = mat(params)
            elif name in ['multicnot', 'multixcnot']:
                # this is for gates that can be applied to arbitrary numbers of
                # qubits but no params, such as multicnot
                matrix = mat(n_wires)
            else:
                # this is for gates that can be applied to arbitrary numbers of
                # qubits such as multirz
                matrix = mat(params, n_wires)

        else:
            matrix = mat

        if inverse:
            matrix = matrix.conj()
            if matrix.dim() == 3:
                matrix = matrix.permute(0, 2, 1)
            else:
                matrix = matrix.permute(1, 0)

        state = q_device.states
        if method == 'einsum':
            q_device.states = apply_unitary_einsum(state, matrix, wires)
        elif method == 'bmm':
            q_device.states = apply_unitary_bmm(state, matrix, wires)


def rx_matrix(params):
    theta = params.type(C_DTYPE)
    """
    Seems to be a pytorch bug. Have to explicitly cast the theta to a 
    complex number. If directly theta = params, then get error:
    
    allow_unreachable=True, accumulate_grad=True)  # allow_unreachable flag
    RuntimeError: Expected isFloatingType(grad.scalar_type()) || 
    (input_is_complex == grad_is_complex) to be true, but got false.  
    (Could this error message be improved?  
    If so, please report an enhancement request to PyTorch.)
        
    """
    co = torch.cos(theta / 2)
    jsi = 1j * torch.sin(-theta / 2)

    return torch.stack([torch.cat([co, jsi], dim=-1),
                        torch.cat([jsi, co], dim=-1)], dim=-2).squeeze(0)


def ry_matrix(params):
    theta = params.type(C_DTYPE)

    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    return torch.stack([torch.cat([co, -si], dim=-1),
                        torch.cat([si, co], dim=-1)], dim=-2).squeeze(0)


def rz_matrix(params):
    theta = params.type(C_DTYPE)
    p = torch.exp(-0.5j * theta)

    return torch.stack([torch.cat([p, torch.zeros(p.shape,
                                                  device=params.device)],
                                  dim=-1),
                        torch.cat([torch.zeros(p.shape, device=params.device),
                                   torch.conj(p)], dim=-1)],
                       dim=-2).squeeze(0)


def phaseshift_matrix(params):
    phi = params.type(C_DTYPE)
    p = torch.exp(1j * phi)

    return torch.stack([
        torch.cat([
            torch.ones(p.shape, device=params.device),
            torch.zeros(p.shape, device=params.device)], dim=-1),
        torch.cat([
            torch.zeros(p.shape, device=params.device),
            p], dim=-1)],
        dim=-2).squeeze(0)


def rot_matrix(params):
    phi = params[:, 0].unsqueeze(dim=-1).type(C_DTYPE)
    theta = params[:, 1].unsqueeze(dim=-1).type(C_DTYPE)
    omega = params[:, 2].unsqueeze(dim=-1).type(C_DTYPE)

    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    return torch.stack([
        torch.cat([
            torch.exp(-0.5j * (phi + omega)) * co,
            -torch.exp(0.5j * (phi - omega)) * si], dim=-1),
        torch.cat([
            torch.exp(-0.5j * (phi - omega)) * si,
            torch.exp(0.5j * (phi + omega)) * co], dim=-1)],
        dim=-2).squeeze(0)


def multirz_eigvals(params, n_wires):
    theta = params.type(C_DTYPE)
    return torch.exp(-1j * theta / 2 * torch.tensor(pauli_eigs(n_wires)).to(
        params.device))


def multirz_matrix(params, n_wires):
    # torch diagonal not available for complex number
    eigvals = multirz_eigvals(params, n_wires)
    dia = diag(eigvals)
    return dia.squeeze(0)


def crx_matrix(params):
    theta = params.type(C_DTYPE)
    co = torch.cos(theta / 2)
    jsi = 1j * torch.sin(-theta / 2)

    matrix = torch.tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(co.shape[0], 1, 1)
    matrix[:, 2, 2] = co[:, 0]
    matrix[:, 2, 3] = jsi[:, 0]
    matrix[:, 3, 2] = jsi[:, 0]
    matrix[:, 3, 3] = co[:, 0]

    return matrix.squeeze(0)


def cry_matrix(params):
    theta = params.type(C_DTYPE)
    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    matrix = torch.tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(co.shape[0], 1, 1)
    matrix[:, 2, 2] = co[:, 0]
    matrix[:, 2, 3] = -si[:, 0]
    matrix[:, 3, 2] = si[:, 0]
    matrix[:, 3, 3] = co[:, 0]

    return matrix.squeeze(0)


def crz_matrix(params):
    theta = params.type(C_DTYPE)
    p = torch.exp(-0.5j * theta)

    matrix = torch.tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(p.shape[0], 1, 1)
    matrix[:, 2, 2] = p[:, 0]
    matrix[:, 3, 3] = torch.conj(p[:, 0])

    return matrix.squeeze(0)


def crot_matrix(params):
    phi = params[:, 0].unsqueeze(dim=-1).type(C_DTYPE)
    theta = params[:, 1].unsqueeze(dim=-1).type(C_DTYPE)
    omega = params[:, 2].unsqueeze(dim=-1).type(C_DTYPE)

    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    matrix = torch.tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(phi.shape[0], 1, 1)

    matrix[:, 2, 2] = (torch.exp(-0.5j * (phi + omega)) * co)[:, 0]
    matrix[:, 2, 3] = (-torch.exp(0.5j * (phi - omega)) * si)[:, 0]
    matrix[:, 3, 2] = (torch.exp(-0.5j * (phi - omega)) * si)[:, 0]
    matrix[:, 3, 3] = (torch.exp(0.5j * (phi + omega)) * co)[:, 0]

    return matrix.squeeze(0)


def u1_matrix(params):
    phi = params.type(C_DTYPE)
    p = torch.exp(1j * phi)

    return torch.stack([
        torch.cat([
            torch.ones(p.shape, device=params.device),
            torch.zeros(p.shape, device=params.device)], dim=-1),
        torch.cat([
            torch.zeros(p.shape, device=params.device),
            p], dim=-1)],
        dim=-2).squeeze(0)


def u2_matrix(params):
    phi = params[:, 0].unsqueeze(dim=-1).type(C_DTYPE)
    lam = params[:, 1].unsqueeze(dim=-1).type(C_DTYPE)

    return INV_SQRT2 * torch.stack([
        torch.cat([
            torch.ones(phi.shape, device=params.device),
            -torch.exp(1j * lam)], dim=-1),
        torch.cat([
            torch.exp(1j * phi),
            torch.exp(1j * (phi + lam))], dim=-1)],
        dim=-2).squeeze(0)


def u3_matrix(params):
    theta = params[:, 0].unsqueeze(dim=-1).type(C_DTYPE)
    phi = params[:, 1].unsqueeze(dim=-1).type(C_DTYPE)
    lam = params[:, 2].unsqueeze(dim=-1).type(C_DTYPE)

    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    return torch.stack([
        torch.cat([
            co,
            -si * torch.exp(1j * lam)], dim=-1),
        torch.cat([
            si * torch.exp(1j * phi),
            co * torch.exp(1j * (phi + lam))], dim=-1)],
        dim=-2).squeeze(0)


def qubitunitary_matrix(params):
    matrix = params.squeeze(0)
    try:
        assert matrix.shape[-1] == matrix.shape[-2]
    except AssertionError as err:
        logger.exception(f"Operator must be a square matrix.")
        raise err

    try:
        U = matrix.cpu().detach().numpy()
        if matrix.dim() > 2:
            # batched unitary
            bsz = matrix.shape[0]
            assert np.allclose(np.matmul(U, np.transpose(U.conj(), [0, 2, 1])),
                               np.stack([np.identity(U.shape[-1])] * bsz),
                               atol=1e-5)
        else:
            assert np.allclose(np.matmul(U, np.transpose(U.conj(), [1, 0])),
                               np.identity(U.shape[0]), atol=1e-5)
    except AssertionError as err:
        logger.exception(f"Operator must be unitary.")
        raise err

    return matrix


def qubitunitaryfast_matrix(params):
    return params.squeeze(0)


def qubitunitarystrict_matrix(params):
    params.squeeze(0)
    mat = params
    U, Sigma, V = torch.svd(mat)
    return U.matmul(V)


def multicnot_matrix(n_wires):
    mat = torch.eye(2 ** n_wires, dtype=C_DTYPE)
    mat[-1][-1] = 0
    mat[-2][-2] = 0
    mat[-1][-2] = 1
    mat[-2][-1] = 1

    return mat


def multixcnot_matrix(n_wires):
    # when all control qubits are zero, then the target qubit will flip
    mat = torch.eye(2 ** n_wires, dtype=C_DTYPE)
    mat[0][0] = 0
    mat[1][1] = 0
    mat[0][1] = 1
    mat[1][0] = 1

    return mat


mat_dict = {
    'hadamard': torch.tensor([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]],
                             dtype=C_DTYPE),
    'paulix': torch.tensor([[0, 1], [1, 0]], dtype=C_DTYPE),
    'pauliy': torch.tensor([[0, -1j], [1j, 0]], dtype=C_DTYPE),
    'pauliz': torch.tensor([[1, 0], [0, -1]], dtype=C_DTYPE),
    's': torch.tensor([[1, 0], [0, 1j]], dtype=C_DTYPE),
    't': torch.tensor([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=C_DTYPE),
    'sx': 0.5 * torch.tensor([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]],
                             dtype=C_DTYPE),
    'cnot': torch.tensor([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0]], dtype=C_DTYPE),
    'cz': torch.tensor([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, -1]], dtype=C_DTYPE),
    'cy': torch.tensor([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, -1j],
                        [0, 0, 1j, 0]], dtype=C_DTYPE),
    'swap': torch.tensor([[1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1]], dtype=C_DTYPE),
    'cswap': torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]], dtype=C_DTYPE),
    'toffoli': torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 1, 0]], dtype=C_DTYPE),
    'rx': rx_matrix,
    'ry': ry_matrix,
    'rz': rz_matrix,
    'phaseshift': phaseshift_matrix,
    'rot': rot_matrix,
    'multirz': multirz_matrix,
    'crx': crx_matrix,
    'cry': cry_matrix,
    'crz': crz_matrix,
    'crot': crot_matrix,
    'u1': u1_matrix,
    'u2': u2_matrix,
    'u3': u3_matrix,
    'qubitunitary': qubitunitary_matrix,
    'qubitunitaryfast': qubitunitaryfast_matrix,
    'qubitunitarystrict': qubitunitarystrict_matrix,
    'multicnot': multicnot_matrix,
    'multixcnot': multixcnot_matrix,
}


hadamard = partial(gate_wrapper, 'hadamard', mat_dict['hadamard'], 'bmm')
paulix = partial(gate_wrapper, 'paulix', mat_dict['paulix'], 'bmm')
pauliy = partial(gate_wrapper, 'pauliy', mat_dict['pauliy'], 'bmm')
pauliz = partial(gate_wrapper, 'pauliz', mat_dict['pauliz'], 'bmm')
s = partial(gate_wrapper, 's', mat_dict['s'], 'bmm')
t = partial(gate_wrapper, 't', mat_dict['t'], 'bmm')
sx = partial(gate_wrapper, 'sx', mat_dict['sx'], 'bmm')
cnot = partial(gate_wrapper, 'cnot', mat_dict['cnot'], 'bmm')
cz = partial(gate_wrapper, 'cz', mat_dict['cz'], 'bmm')
cy = partial(gate_wrapper, 'cy', mat_dict['cy'], 'bmm')
rx = partial(gate_wrapper, 'rx', mat_dict['rx'], 'bmm')
ry = partial(gate_wrapper, 'ry', mat_dict['ry'], 'bmm')
rz = partial(gate_wrapper, 'rz', mat_dict['rz'], 'bmm')
swap = partial(gate_wrapper, 'swap', mat_dict['swap'], 'bmm')
cswap = partial(gate_wrapper, 'cswap', mat_dict['cswap'], 'bmm')
toffoli = partial(gate_wrapper, 'toffoli', mat_dict['toffoli'], 'bmm')
phaseshift = partial(gate_wrapper, 'phaseshift', mat_dict['phaseshift'], 'bmm')
rot = partial(gate_wrapper, 'rot', mat_dict['rot'], 'bmm')
multirz = partial(gate_wrapper, 'multirz', mat_dict['multirz'], 'bmm')
crx = partial(gate_wrapper, 'crx', mat_dict['crx'], 'bmm')
cry = partial(gate_wrapper, 'cry', mat_dict['cry'], 'bmm')
crz = partial(gate_wrapper, 'crz', mat_dict['crz'], 'bmm')
crot = partial(gate_wrapper, 'crot', mat_dict['crot'], 'bmm')
u1 = partial(gate_wrapper, 'u1', mat_dict['u1'], 'bmm')
u2 = partial(gate_wrapper, 'u2', mat_dict['u2'], 'bmm')
u3 = partial(gate_wrapper, 'u3', mat_dict['u3'], 'bmm')
qubitunitary = partial(gate_wrapper, 'qubitunitary', mat_dict[
    'qubitunitary'], 'bmm')
qubitunitaryfast = partial(gate_wrapper, 'qubitunitaryfast', mat_dict[
    'qubitunitaryfast'], 'bmm')
qubitunitarystrict = partial(gate_wrapper, 'qubitunitarystrict', mat_dict[
    'qubitunitarystrict'], 'bmm')
multicnot = partial(gate_wrapper, 'multicnot', mat_dict['multicnot'], 'bmm')
multixcnot = partial(gate_wrapper, 'multixcnot', mat_dict['multixcnot'], 'bmm')


h = hadamard
x = paulix
y = pauliy
z = pauliz
cx = cnot
ccnot = toffoli
ccx = toffoli


func_name_dict = {
    'hadamard': hadamard,
    'paulix': paulix,
    'pauliy': pauliy,
    'pauliz': pauliz,
    's': s,
    't': t,
    'sx': sx,
    'cnot': cnot,
    'cz': cz,
    'cy': cy,
    'rx': rx,
    'ry': ry,
    'rz': rz,
    'swap': swap,
    'cswap': cswap,
    'toffoli': toffoli,
    'phaseshift': phaseshift,
    'rot': rot,
    'multirz': multirz,
    'crx': crx,
    'cry': cry,
    'crz': crz,
    'crot': crot,
    'u1': u1,
    'u2': u2,
    'u3': u3,
    'qubitunitary': qubitunitary,
    'qubitunitaryfast': qubitunitaryfast,
    'qubitunitarystrict': qubitunitarystrict,
    'multicnot': multicnot,
    'multixcnot': multixcnot,
    'x': x,
    'y': y,
    'z': z,
    'cx': cx,
    'ccnot': ccnot,
    'ccx': ccx,
}
