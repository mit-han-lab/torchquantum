"""
Measure the torch.permute + torch.bmm V.S. torch.einsum for the
multiplication between statevector and gate unitary matrix
"""
import functools
import torch
import torchquantum as tq
import numpy as np
import pdb
import argparse
import logging

from torchquantum.macro import ABC, ABC_ARRAY
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_unitary_einsum(state, mat, wires):
    device_wires = wires

    total_wires = len(state.shape) - 1

    if len(mat.shape) > 2:
        is_batch_unitary = True
        bsz = mat.shape[0]
        shape_extension = [bsz]
        try:
            assert state.shape[0] == bsz
        except AssertionError as err:
            logger.exception(f"Batch size of Quantum Device must be the same "
                             f"with that of gate unitary matrix")
            raise err

    else:
        is_batch_unitary = False
        shape_extension = []

    mat = mat.view(shape_extension + [2] * len(device_wires) * 2)

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


def apply_unitary_bmm(state, mat, wires, contiguous=True):
    device_wires = wires

    if len(mat.shape) > 2:
        is_batch_unitary = True
        bsz = mat.shape[0]
        try:
            assert state.shape[0] == bsz
        except AssertionError as err:
            logger.exception(f"Batch size of Quantum Device must be the same "
                             f"with that of gate unitary matrix")
            raise err

    else:
        is_batch_unitary = False

    devices_dims = [x + 1 for x in device_wires]
    permute_to = list(range(state.dim()))
    for d in sorted(devices_dims, reverse=True):
        del permute_to[d]
    permute_to += devices_dims
    permute_back = list(np.argsort(permute_to))
    original_shape = state.shape
    permuted = state.permute(permute_to)

    if contiguous:
        permuted = permuted.contiguous().view(
            [original_shape[0], -1, mat.shape[-1]])
    else:
        permuted = permuted.reshape(
            [original_shape[0], -1, mat.shape[-1]])

    if is_batch_unitary:
        mat = mat.permute(0, 2, 1)
    else:
        mat = mat.permute(1, 0)

    mat = mat.contiguous() if contiguous else mat

    new_state = permuted.matmul(mat).view(original_shape).permute(
        permute_back)

    new_state = new_state.contiguous() if contiguous else new_state

    return new_state


def apply_unitary_bmm_no_mat_transpose(state, mat, wires, contiguous=True):
    device_wires = wires

    if len(mat.shape) > 2:
        bsz = mat.shape[0]
        try:
            assert state.shape[0] == bsz
        except AssertionError as err:
            logger.exception(f"Batch size of Quantum Device must be the same "
                             f"with that of gate unitary matrix")
            raise err

    devices_dims = [x + 1 for x in device_wires]
    permute_to = list(range(state.dim()))
    for d in sorted(devices_dims, reverse=True):
        del permute_to[d]
    permute_to = permute_to[:1] + devices_dims + permute_to[1:]
    permute_back = list(np.argsort(permute_to))
    original_shape = state.shape
    permuted = state.permute(permute_to)

    if contiguous:
        permuted = permuted.contiguous().view(
            [original_shape[0], mat.shape[-1], -1])
    else:
        permuted = permuted.reshape(
            [original_shape[0], mat.shape[-1], -1])

    new_state = mat.matmul(permuted).view(original_shape).permute(
        permute_back)

    new_state = new_state.contiguous() if contiguous else new_state

    return new_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--device', type=str, default='gpu', help='cpu or gpu')
    args = parser.parse_args()

    if args.pdb:
        pdb.set_trace()

    batch_size = 512
    n_wires = 12
    dims = [batch_size] + [2] * n_wires
    device = torch.device('cuda') if args.device == 'gpu' else torch.device(
        'cpu')
    state_in = torch.randn(*dims, dtype=tq.C_DTYPE, device=device)

    run_times = 1000

    is_batch_matrix = True

    for n_gate_wires in tqdm(range(1, 10)):
        if is_batch_matrix:
            matrix = torch.randn(batch_size, 2 ** n_gate_wires,
                                 2 ** n_gate_wires,
                                 dtype=tq.C_DTYPE, device=device)
        else:
            matrix = torch.randn(2 ** n_gate_wires, 2 ** n_gate_wires,
                                 dtype=tq.C_DTYPE, device=device)

        wires_in = list(np.random.choice(n_wires, n_gate_wires, replace=False))

        res0 = apply_unitary_bmm(state_in, matrix, wires_in, contiguous=True)
        res1 = apply_unitary_einsum(state_in, matrix, wires_in)
        res2 = apply_unitary_bmm_no_mat_transpose(state_in, matrix,
                                                  wires_in, contiguous=False)
        diff0 = torch.abs(res0-res1).mean()
        diff1 = torch.abs(res1-res2).mean()
        diff2 = torch.abs(res2-res0).mean()

        logger.warning(f"Difference: {diff0, diff1, diff2}")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # dry run:
        for _ in range(5):
            res = apply_unitary_einsum(state_in, matrix, wires_in)
        start.record()
        for _ in range(run_times):
            res = apply_unitary_einsum(state_in, matrix, wires_in)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        logger.info(f"einsum: {n_gate_wires}:"
                    f" {start.elapsed_time(end) / run_times}")

        # dry run:
        for _ in range(5):
            res = apply_unitary_bmm(state_in, matrix, wires_in,
                                    contiguous=True)
        start.record()
        for _ in range(run_times):
            res = apply_unitary_bmm(state_in, matrix, wires_in,
                                    contiguous=True)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        logger.info(f"bmm contiguous: {n_gate_wires}:"
                    f" {start.elapsed_time(end) / run_times}")

        # dry run:
        for _ in range(5):
            res = apply_unitary_bmm(state_in, matrix, wires_in,
                                    contiguous=False)
        start.record()
        for _ in range(run_times):
            res = apply_unitary_bmm(state_in, matrix, wires_in,
                                    contiguous=False)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        logger.info(f"bmm no contiguous: {n_gate_wires}:"
                    f" {start.elapsed_time(end) / run_times}")

        # dry run:
        for _ in range(5):
            res = apply_unitary_bmm_no_mat_transpose(
                state_in, matrix, wires_in, contiguous=False)
        start.record()
        for _ in range(run_times):
            res = apply_unitary_bmm_no_mat_transpose(
                state_in, matrix, wires_in, contiguous=False)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        logger.info(f"bmm no contiguous no mat transpose: {n_gate_wires}:"
                    f" {start.elapsed_time(end) / run_times}")

        # dry run:
        for _ in range(5):
            res = apply_unitary_bmm_no_mat_transpose(
                state_in, matrix, wires_in, contiguous=True)
        start.record()
        for _ in range(run_times):
            res = apply_unitary_bmm_no_mat_transpose(
                state_in, matrix, wires_in, contiguous=True)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        logger.info(f"bmm contiguous no mat transpose: {n_gate_wires}:"
                    f" {start.elapsed_time(end) / run_times}")
