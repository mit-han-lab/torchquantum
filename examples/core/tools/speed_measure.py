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

from functools import partial
from typing import Callable
from torchquantum.macro import C_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from torchquantum.utils import pauli_eigs, diag
from torchpack.utils.logging import logger
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


def apply_unitary_bmm(state, mat, wires):
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

    devices_dims = [x + 1 for x in device_wires]
    permute_to = list(range(state.dim()))
    for d in sorted(devices_dims, reverse=True):
        del permute_to[d]
    permute_to += devices_dims
    permute_back = list(np.argsort(permute_to))
    original_shape = state.shape

    permuted = state.permute(permute_to).contiguous().view(
        [original_shape[0], -1, mat.shape[-1]])

    if is_batch_unitary:
        mat = mat.permute(0, 2, 1).contiguous()
    else:
        mat = mat.transpose().contiguous()

    new_state = permuted.matmul(mat).view(original_shape).permute(
        permute_back)

    return new_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--device', type=str, default='gpu', help='cpu or gpu')
    args = parser.parse_args()

    if args.pdb:
        pdb.set_trace()

    bsz = 64
    n_wires = 12
    dims = [bsz] + [2] * n_wires
    device = torch.device('cuda') if args.device == 'gpu' else torch.device(
        'cpu')
    state = torch.randn(*dims, dtype=tq.C_DTYPE, device=device)

    run_times = 10000

    for n_gate_wires in tqdm(range(8, 12)):
        mat = torch.randn(bsz, 2 ** n_gate_wires, 2 ** n_gate_wires,
                          dtype=tq.C_DTYPE, device=device)

        wires = list(np.random.choice(n_wires, n_gate_wires, replace=False))

        res0 = apply_unitary_bmm(state, mat, wires)
        res1 = apply_unitary_einsum(state, mat, wires)
        diff = torch.abs(res1-res0).max()
        assert diff < 1e-3

        # diff = torch.abs(res1-res0).max()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # dry run:
        for _ in range(5):
            res = apply_unitary_einsum(state, mat, wires)
        start.record()
        for _ in range(run_times):
            res = apply_unitary_einsum(state, mat, wires)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        logger.info(f"einsum: {n_gate_wires}:"
                    f" {start.elapsed_time(end) / run_times}")

        # dry run:
        for _ in range(5):
            res = apply_unitary_bmm(state, mat, wires)
        start.record()
        for _ in range(run_times):
            res = apply_unitary_bmm(state, mat, wires)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        logger.info(f"bmm: {n_gate_wires}:"
                    f" {start.elapsed_time(end) / run_times}")
