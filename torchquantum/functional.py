import functools
import torch
import numpy as np

import torchquantum as tq

from typing import Callable, Union, Optional, List, Dict
from .macro import C_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from .utils import pauli_eigs, diag
from torchpack.utils.logging import logger
from torchquantum.utils import normalize_statevector

__all__ = [
    'func_name_dict',
    'mat_dict',
    'apply_unitary_einsum',
    'apply_unitary_bmm',
    'hadamard',
    'shadamard',
    'paulix',
    'pauliy',
    'pauliz',
    'i',
    's',
    't',
    'sx',
    'cnot',
    'cz',
    'cy',
    'swap',
    'sswap',
    'cswap',
    'toffoli',
    'multicnot',
    'multixcnot',
    'rx',
    'ry',
    'rz',
    'rxx',
    'ryy',
    'rzz',
    'rzx',
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
    'cu1',
    'cu2',
    'cu3',
    'qubitunitary',
    'qubitunitaryfast',
    'qubitunitarystrict',
    'single_excitation',
    'h',
    'sh',
    'x',
    'y',
    'z',
    'xx',
    'yy',
    'zz',
    'zx',
    'cx',
    'ccnot',
    'ccx',
    'u',
    'cu',
    'p',
    'cp',
    'cr',
    'cphase',
    'reset',
]


def apply_unitary_einsum(state, mat, wires):
    """Apply the unitary to the statevector using torch.einsum method.

    Args:
        state (torch.Tensor): The statevector.
        mat (torch.Tensor): The unitary matrix of the operation.
        wires (int or List[int]): Which qubit the operation is applied to.

    Returns:
        torch.Tensor: The new statevector.

    """
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
    """Apply the unitary to the statevector using torch.bmm method.

        Args:
            state (torch.Tensor): The statevector.
            mat (torch.Tensor): The unitary matrix of the operation.
            wires (int or List[int]): Which qubit the operation is applied to.

        Returns:
            torch.Tensor: The new statevector.

        """
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
    """Perform the phaseshift gate.

    Args:
        name (str): The name of the operation.
        mat (torch.Tensor): The unitary matrix of the gate.
        method (str): 'bmm' or 'einsum' to compute matrix vector
            multiplication.
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
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
            elif name in ['multirz']:
                # this is for gates that can be applied to arbitrary numbers of
                # qubits such as multirz
                matrix = mat(params, n_wires)
            else:
                matrix = mat(params)

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


def reset(q_device: tq.QuantumDevice, wires, inverse=False):
    # reset the target qubits to 0, non-unitary operation
    state = q_device.states

    wires = [wires] if isinstance(wires, int) else wires

    for wire in wires:
        devices_dim = wire + 1
        permute_to = list(range(state.dim()))
        del permute_to[devices_dim]
        permute_to += [devices_dim]
        permute_back = list(np.argsort(permute_to))

        # permute the target wire to the last dim
        permuted = state.permute(permute_to)

        # reset the state
        permuted[..., 1] = 0

        # permute back
        state = state.permute(permute_back)

    # normalize the magnitude of states
    q_device.states = normalize_statevector(q_device.states)


def rx_matrix(params: torch.Tensor) -> torch.Tensor:
    """Compute unitary matrix for rx gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
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


def ry_matrix(params: torch.Tensor) -> torch.Tensor:
    """Compute unitary matrix for ry gate.

    Args:
        params: The rotation angle.

    Returns:
        The computed unitary matrix.

    """
    theta = params.type(C_DTYPE)

    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    return torch.stack([torch.cat([co, -si], dim=-1),
                        torch.cat([si, co], dim=-1)], dim=-2).squeeze(0)


def rz_matrix(params: torch.Tensor) -> torch.Tensor:
    """Compute unitary matrix for rz gate.

    Args:
        params: The rotation angle.

    Returns:
        The computed unitary matrix.

    """
    theta = params.type(C_DTYPE)
    exp = torch.exp(-0.5j * theta)

    return torch.stack([torch.cat([exp, torch.zeros(exp.shape,
                                                    device=params.device)],
                                  dim=-1),
                        torch.cat([torch.zeros(exp.shape,
                                               device=params.device),
                                   torch.conj(exp)], dim=-1)],
                       dim=-2).squeeze(0)


def phaseshift_matrix(params):
    phi = params.type(C_DTYPE)
    exp = torch.exp(1j * phi)

    return torch.stack([
        torch.cat([
            torch.ones(exp.shape, device=params.device),
            torch.zeros(exp.shape, device=params.device)], dim=-1),
        torch.cat([
            torch.zeros(exp.shape, device=params.device),
            exp], dim=-1)],
        dim=-2).squeeze(0)


def rot_matrix(params):
    """Compute unitary matrix for rot gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
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
    """Compute eigenvalue for multiqubit RZ gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed eigenvalues.

    """
    theta = params.type(C_DTYPE)
    return torch.exp(-1j * theta / 2 * torch.tensor(pauli_eigs(n_wires)).to(
        params.device))


def multirz_matrix(params, n_wires):
    """Compute unitary matrix for multiqubit RZ gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    # torch diagonal not available for complex number
    eigvals = multirz_eigvals(params, n_wires)
    dia = diag(eigvals)
    return dia.squeeze(0)


def rxx_matrix(params):
    """Compute unitary matrix for RXX gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """

    theta = params.type(C_DTYPE)
    co = torch.cos(theta / 2)
    jsi = 1j * torch.sin(theta / 2)

    matrix = torch.tensor([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(co.shape[0], 1, 1)

    matrix[:, 0, 0] = co[:, 0]
    matrix[:, 1, 1] = co[:, 0]
    matrix[:, 2, 2] = co[:, 0]
    matrix[:, 3, 3] = co[:, 0]

    matrix[:, 0, 3] = -jsi[:, 0]
    matrix[:, 1, 2] = -jsi[:, 0]
    matrix[:, 2, 1] = -jsi[:, 0]
    matrix[:, 3, 0] = -jsi[:, 0]

    return matrix.squeeze(0)


def ryy_matrix(params):
    """Compute unitary matrix for RYY gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    theta = params.type(C_DTYPE)
    co = torch.cos(theta / 2)
    jsi = 1j * torch.sin(theta / 2)

    matrix = torch.tensor([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(co.shape[0], 1, 1)

    matrix[:, 0, 0] = co[:, 0]
    matrix[:, 1, 1] = co[:, 0]
    matrix[:, 2, 2] = co[:, 0]
    matrix[:, 3, 3] = co[:, 0]

    matrix[:, 0, 3] = jsi[:, 0]
    matrix[:, 1, 2] = -jsi[:, 0]
    matrix[:, 2, 1] = -jsi[:, 0]
    matrix[:, 3, 0] = jsi[:, 0]

    return matrix.squeeze(0)


def rzz_matrix(params):
    """Compute unitary matrix for RZZ gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    theta = params.type(C_DTYPE)
    exp = torch.exp(-0.5j * theta)
    conj_exp = torch.conj(exp)

    matrix = torch.tensor([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(exp.shape[0], 1, 1)

    matrix[:, 0, 0] = exp[:, 0]
    matrix[:, 1, 1] = conj_exp[:, 0]
    matrix[:, 2, 2] = conj_exp[:, 0]
    matrix[:, 3, 3] = exp[:, 0]

    return matrix.squeeze(0)


def rzx_matrix(params):
    """Compute unitary matrix for RZX gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    theta = params.type(C_DTYPE)
    co = torch.cos(theta / 2)
    jsi = 1j * torch.sin(theta / 2)

    matrix = torch.tensor([[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(co.shape[0], 1, 1)

    matrix[:, 0, 0] = co[:, 0]
    matrix[:, 0, 1] = -jsi[:, 0]

    matrix[:, 1, 0] = -jsi[:, 0]
    matrix[:, 1, 1] = co[:, 0]

    matrix[:, 2, 2] = co[:, 0]
    matrix[:, 2, 3] = jsi[:, 0]

    matrix[:, 3, 2] = jsi[:, 0]
    matrix[:, 3, 3] = co[:, 0]

    return matrix.squeeze(0)


def crx_matrix(params):
    """Compute unitary matrix for CRX gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
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
    """Compute unitary matrix for CRY gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
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
    """Compute unitary matrix for CRZ gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    theta = params.type(C_DTYPE)
    exp = torch.exp(-0.5j * theta)

    matrix = torch.tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(exp.shape[0], 1, 1)
    matrix[:, 2, 2] = exp[:, 0]
    matrix[:, 3, 3] = torch.conj(exp[:, 0])

    return matrix.squeeze(0)


def crot_matrix(params):
    """Compute unitary matrix for CRot gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    phi = params[:, 0].type(C_DTYPE)
    theta = params[:, 1].type(C_DTYPE)
    omega = params[:, 2].type(C_DTYPE)

    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    matrix = torch.tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(phi.shape[0], 1, 1)

    matrix[:, 2, 2] = torch.exp(-0.5j * (phi + omega)) * co
    matrix[:, 2, 3] = -torch.exp(0.5j * (phi - omega)) * si
    matrix[:, 3, 2] = torch.exp(-0.5j * (phi - omega)) * si
    matrix[:, 3, 3] = torch.exp(0.5j * (phi + omega)) * co

    return matrix.squeeze(0)


def u1_matrix(params):
    """Compute unitary matrix for U1 gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    phi = params.type(C_DTYPE)
    exp = torch.exp(1j * phi)

    return torch.stack([
        torch.cat([
            torch.ones(exp.shape, device=params.device),
            torch.zeros(exp.shape, device=params.device)], dim=-1),
        torch.cat([
            torch.zeros(exp.shape, device=params.device),
            exp], dim=-1)],
        dim=-2).squeeze(0)


def cu1_matrix(params):
    """Compute unitary matrix for CU1 gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    phi = params.type(C_DTYPE)
    exp = torch.exp(1j * phi)

    matrix = torch.tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(phi.shape[0], 1, 1)

    matrix[:, 3, 3] = exp

    return matrix.squeeze(0)


def u2_matrix(params):
    """Compute unitary matrix for U2 gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
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


def cu2_matrix(params):
    """Compute unitary matrix for CU2 gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    phi = params[:, 0].unsqueeze(dim=-1).type(C_DTYPE)
    lam = params[:, 1].unsqueeze(dim=-1).type(C_DTYPE)

    matrix = torch.tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(phi.shape[0], 1, 1)

    matrix[:, 2, 3] = -torch.exp(1j * lam)
    matrix[:, 3, 2] = torch.exp(1j * phi)
    matrix[:, 3, 3] = torch.exp(1j * (phi + lam))

    return matrix.squeeze(0)


def u3_matrix(params):
    """Compute unitary matrix for U3 gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
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


def cu3_matrix(params):
    """Compute unitary matrix for CU3 gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    theta = params[:, 0].unsqueeze(dim=-1).type(C_DTYPE)
    phi = params[:, 1].unsqueeze(dim=-1).type(C_DTYPE)
    lam = params[:, 2].unsqueeze(dim=-1).type(C_DTYPE)

    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    matrix = torch.tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(phi.shape[0], 1, 1)

    matrix[:, 2, 2] = co
    matrix[:, 2, 3] = -si * torch.exp(1j * lam)
    matrix[:, 3, 2] = si * torch.exp(1j * phi)
    matrix[:, 3, 3] = co * torch.exp(1j * (phi + lam))

    return matrix.squeeze(0)


def qubitunitary_matrix(params):
    """Compute unitary matrix for Qubitunitary gate.

    Args:
        params (torch.Tensor): The unitary matrix.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
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
    """Compute unitary matrix for Qubitunitary fast gate.

    Args:
        params (torch.Tensor): The unitary matrix.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    return params.squeeze(0)


def qubitunitarystrict_matrix(params):
    """Compute unitary matrix for Qubitunitary strict gate.
        Strictly be the unitary.

    Args:
        params (torch.Tensor): The unitary matrix.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    params.squeeze(0)
    mat = params
    U, Sigma, V = torch.svd(mat)
    return U.matmul(V)


def multicnot_matrix(n_wires):
    """Compute unitary matrix for Multi qubit CNOT gate.

    Args:
        n_wires (int): The number of wires.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    mat = torch.eye(2 ** n_wires, dtype=C_DTYPE)
    mat[-1][-1] = 0
    mat[-2][-2] = 0
    mat[-1][-2] = 1
    mat[-2][-1] = 1

    return mat


def multixcnot_matrix(n_wires):
    """Compute unitary matrix for Multi qubit XCNOT gate.

    Args:
        params (torch.Tensor): The unitary matrix.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    # when all control qubits are zero, then the target qubit will flip
    mat = torch.eye(2 ** n_wires, dtype=C_DTYPE)
    mat[0][0] = 0
    mat[1][1] = 0
    mat[0][1] = 1
    mat[1][0] = 1

    return mat


def single_excitation_matrix(params):
    """Compute unitary matrix for single excitation gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    theta = params.type(C_DTYPE)
    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    matrix = torch.tensor([[1, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1]], dtype=C_DTYPE, device=params.device
                          ).unsqueeze(0).repeat(theta.shape[0], 1, 1)

    matrix[:, 1, 1] = co
    matrix[:, 1, 2] = -si
    matrix[:, 2, 1] = si
    matrix[:, 2, 2] = co

    return matrix.squeeze(0)


mat_dict = {
    'hadamard': torch.tensor([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]],
                             dtype=C_DTYPE),
    'shadamard': torch.tensor([[np.cos(np.pi / 8), -np.sin(np.pi / 8)],
                               [np.sin(np.pi / 8), np.cos(np.pi / 8)]],
                              dtype=C_DTYPE),
    'paulix': torch.tensor([[0, 1], [1, 0]], dtype=C_DTYPE),
    'pauliy': torch.tensor([[0, -1j], [1j, 0]], dtype=C_DTYPE),
    'pauliz': torch.tensor([[1, 0], [0, -1]], dtype=C_DTYPE),
    'i': torch.tensor([[1, 0], [0, 1]], dtype=C_DTYPE),
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
    'sswap': torch.tensor([[1, 0, 0, 0],
                           [0, (1 + 1j) / 2, (1 - 1j) / 2, 0],
                           [0, (1 - 1j) / 2, (1 + 1j) / 2, 0],
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
    'rxx': rxx_matrix,
    'ryy': ryy_matrix,
    'rzz': rzz_matrix,
    'rzx': rzx_matrix,
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
    'cu1': cu1_matrix,
    'cu2': cu2_matrix,
    'cu3': cu3_matrix,
    'qubitunitary': qubitunitary_matrix,
    'qubitunitaryfast': qubitunitaryfast_matrix,
    'qubitunitarystrict': qubitunitarystrict_matrix,
    'multicnot': multicnot_matrix,
    'multixcnot': multixcnot_matrix,
    'single_excitation': single_excitation_matrix,
}


def hadamard(q_device: tq.QuantumDevice,
             wires: Union[List[int], int],
             params: torch.Tensor = None,
             n_wires: int = None,
             static: bool = False,
             parent_graph = None,
             inverse: bool = False,
             comp_method: str = 'bmm'):
    """Perform the hadamard gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to. 
            Default to None.
        static (bool, optional): Whether use static mode computation. 
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of 
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform 
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'hadamard'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def shadamard(q_device,
              wires,
              params=None,
              n_wires=None,
              static=False,
              parent_graph=None,
              inverse=False,
              comp_method='bmm'):
    """Perform the shadamard gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'shadamard'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def paulix(q_device,
           wires,
           params=None,
           n_wires=None,
           static=False,
           parent_graph=None,
           inverse=False,
           comp_method='bmm'):
    """Perform the Pauli X gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'paulix'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def pauliy(q_device,
           wires,
           params=None,
           n_wires=None,
           static=False,
           parent_graph=None,
           inverse=False,
           comp_method='bmm'):
    """Perform the Pauli Y gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'pauliy'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def pauliz(q_device,
           wires,
           params=None,
           n_wires=None,
           static=False,
           parent_graph=None,
           inverse=False,
           comp_method='bmm'):
    """Perform the Pauli Z gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'pauliz'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def i(q_device,
      wires,
      params=None,
      n_wires=None,
      static=False,
      parent_graph=None,
      inverse=False,
      comp_method='bmm'):
    """Perform the I gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'i'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def s(q_device,
      wires,
      params=None,
      n_wires=None,
      static=False,
      parent_graph=None,
      inverse=False,
      comp_method='bmm'):
    """Perform the s gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 's'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def t(q_device,
      wires,
      params=None,
      n_wires=None,
      static=False,
      parent_graph=None,
      inverse=False,
      comp_method='bmm'):
    """Perform the t gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 't'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def sx(q_device,
       wires,
       params=None,
       n_wires=None,
       static=False,
       parent_graph=None,
       inverse=False,
       comp_method='bmm'):
    """Perform the sx gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'sx'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def cnot(q_device,
         wires,
         params=None,
         n_wires=None,
         static=False,
         parent_graph=None,
         inverse=False,
         comp_method='bmm'):
    """Perform the cnot gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'cnot'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def cz(q_device,
       wires,
       params=None,
       n_wires=None,
       static=False,
       parent_graph=None,
       inverse=False,
       comp_method='bmm'):
    """Perform the cz gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'cz'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def cy(q_device,
       wires,
       params=None,
       n_wires=None,
       static=False,
       parent_graph=None,
       inverse=False,
       comp_method='bmm'):
    """Perform the cy gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'cy'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def rx(q_device,
       wires,
       params=None,
       n_wires=None,
       static=False,
       parent_graph=None,
       inverse=False,
       comp_method='bmm'):
    """Perform the rx gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'rx'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def ry(q_device,
       wires,
       params=None,
       n_wires=None,
       static=False,
       parent_graph=None,
       inverse=False,
       comp_method='bmm'):
    """Perform the ry gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'ry'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def rz(q_device,
       wires,
       params=None,
       n_wires=None,
       static=False,
       parent_graph=None,
       inverse=False,
       comp_method='bmm'):
    """Perform the rz gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'rz'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def rxx(q_device,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
        comp_method='bmm'):
    """Perform the rxx gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'rxx'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def ryy(q_device,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
        comp_method='bmm'):
    """Perform the ryy gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'ryy'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def rzz(q_device,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
        comp_method='bmm'):
    """Perform the rzz gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'rzz'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def rzx(q_device,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
        comp_method='bmm'):
    """Perform the rzx gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'rzx'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def swap(q_device,
         wires,
         params=None,
         n_wires=None,
         static=False,
         parent_graph=None,
         inverse=False,
         comp_method='bmm'):
    """Perform the swap gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'swap'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def sswap(q_device,
          wires,
          params=None,
          n_wires=None,
          static=False,
          parent_graph=None,
          inverse=False,
          comp_method='bmm'):
    """Perform the sswap gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'sswap'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def cswap(q_device,
          wires,
          params=None,
          n_wires=None,
          static=False,
          parent_graph=None,
          inverse=False,
          comp_method='bmm'):
    """Perform the cswap gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'cswap'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def toffoli(q_device,
            wires,
            params=None,
            n_wires=None,
            static=False,
            parent_graph=None,
            inverse=False,
            comp_method='bmm'):
    """Perform the toffoli gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'toffoli'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def phaseshift(q_device,
               wires,
               params=None,
               n_wires=None,
               static=False,
               parent_graph=None,
               inverse=False,
               comp_method='bmm'):
    """Perform the phaseshift gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'phaseshift'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def rot(q_device,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
        comp_method='bmm'):
    """Perform the rot gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'rot'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def multirz(q_device,
            wires,
            params=None,
            n_wires=None,
            static=False,
            parent_graph=None,
            inverse=False,
            comp_method='bmm'):
    """Perform the multi qubit RZ gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'multirz'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def crx(q_device,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
        comp_method='bmm'):
    """Perform the crx gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'crx'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def cry(q_device,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
        comp_method='bmm'):
    """Perform the cry gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'cry'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def crz(q_device,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
        comp_method='bmm'):
    """Perform the crz gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'crz'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def crot(q_device,
         wires,
         params=None,
         n_wires=None,
         static=False,
         parent_graph=None,
         inverse=False,
         comp_method='bmm'):
    """Perform the crot gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'crot'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def u1(q_device,
         wires,
         params=None,
         n_wires=None,
         static=False,
         parent_graph=None,
         inverse=False,
         comp_method='bmm'):
    """Perform the u1 gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'u1'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def u2(q_device,
       wires,
       params=None,
       n_wires=None,
       static=False,
       parent_graph=None,
       inverse=False,
       comp_method='bmm'):
    """Perform the u2 gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'u2'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def u3(q_device,
       wires,
       params=None,
       n_wires=None,
       static=False,
       parent_graph=None,
       inverse=False,
       comp_method='bmm'):
    """Perform the u3 gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'u3'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def cu1(q_device,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
        comp_method='bmm'):
    """Perform the cu1 gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'cu1'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def cu2(q_device,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
        comp_method='bmm'):
    """Perform the cu2 gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'cu2'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def cu3(q_device,
        wires,
        params=None,
        n_wires=None,
        static=False,
        parent_graph=None,
        inverse=False,
        comp_method='bmm'):
    """Perform the cu3 gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'cu3'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )



def qubitunitary(q_device,
                 wires,
                 params=None,
                 n_wires=None,
                 static=False,
                 parent_graph=None,
                 inverse=False,
                 comp_method='bmm'):
    """Perform the qubitunitary gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'qubitunitary'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def qubitunitaryfast(q_device,
                     wires,
                     params=None,
                     n_wires=None,
                     static=False,
                     parent_graph=None,
                     inverse=False,
                     comp_method='bmm'):
    """Perform the qubitunitaryfast gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'qubitunitaryfast'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def qubitunitarystrict(q_device,
                       wires,
                       params=None,
                       n_wires=None,
                       static=False,
                       parent_graph=None,
                       inverse=False,
                       comp_method='bmm'):
    """Perform the qubitunitarystrict = gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'qubitunitarystrict'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def multicnot(q_device,
              wires,
              params=None,
              n_wires=None,
              static=False,
              parent_graph=None,
              inverse=False,
              comp_method='bmm'):
    """Perform the multi qubit cnot gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'multicnot'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def multixcnot(q_device,
              wires,
              params=None,
              n_wires=None,
              static=False,
              parent_graph=None,
              inverse=False,
              comp_method='bmm'):
    """Perform the multi qubit xcnot gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'multixcnot'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


def single_excitation(q_device,
                      wires,
                      params=None,
                      n_wires=None,
                      static=False,
                      parent_graph=None,
                      inverse=False,
                      comp_method='bmm'):
    """Perform the single excitation gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = 'single_excitation'
    mat = mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse
    )


h = hadamard
sh = shadamard
x = paulix
y = pauliy
z = pauliz
xx = rxx
yy = ryy
zz = rzz
zx = rzx
cx = cnot
ccnot = toffoli
ccx = toffoli
u = u3
cu = cu3
p = phaseshift
cp = cu1
cr = cu1
cphase = cu1

func_name_dict = {
    'hadamard': hadamard,
    'sh': shadamard,
    'paulix': paulix,
    'pauliy': pauliy,
    'pauliz': pauliz,
    'i': i,
    's': s,
    't': t,
    'sx': sx,
    'cnot': cnot,
    'cz': cz,
    'cy': cy,
    'rx': rx,
    'ry': ry,
    'rz': rz,
    'rxx': rxx,
    'xx': xx,
    'ryy': ryy,
    'yy': yy,
    'rzz': rzz,
    'zz': zz,
    'rzx': rzx,
    'zx': zx,
    'swap': swap,
    'sswap': sswap,
    'cswap': cswap,
    'toffoli': toffoli,
    'phaseshift': phaseshift,
    'p': p,
    'cp': cp,
    'rot': rot,
    'multirz': multirz,
    'crx': crx,
    'cry': cry,
    'crz': crz,
    'crot': crot,
    'u1': u1,
    'u2': u2,
    'u3': u3,
    'u': u,
    'cu1': cu1,
    'cphase': cphase,
    'cr': cr,
    'cu2': cu2,
    'cu3': cu3,
    'cu': cu,
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
    'reset': reset,
}
