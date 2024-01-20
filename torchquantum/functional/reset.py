import functools
import torch
import numpy as np

from typing import Callable, Union, Optional, List, Dict, TYPE_CHECKING
from ..macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from ..util.utils import pauli_eigs, diag
from torchpack.utils.logging import logger
from torchquantum.util import normalize_statevector

from .gate_wrapper import gate_wrapper, apply_unitary_einsum, apply_unitary_bmm

if TYPE_CHECKING:
    from torchquantum.device import QuantumDevice
else:
    QuantumDevice = None


def reset(q_device: QuantumDevice, wires, inverse=False):
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
