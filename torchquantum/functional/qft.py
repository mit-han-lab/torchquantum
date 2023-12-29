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


def qft_matrix(n_wires):
    """Compute unitary matrix for QFT.

    Args:
        n_wires: the number of qubits
    """
    dimension = 2**n_wires
    mat = torch.zeros((dimension, dimension), dtype=torch.complex64)
    omega = np.exp(2 * np.pi * 1j / dimension)

    for m in range(dimension):
        for n in range(dimension):
            mat[m, n] = omega ** (m * n)
    mat = mat / np.sqrt(dimension)
    return mat


_qft_mat_dict = {
    "qft": qft_matrix,
}


def qft(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    name = "qft"
    if n_wires == None:
        wires = [wires] if isinstance(wires, int) else wires
        n_wires = len(wires)

    mat = _qft_mat_dict[name]
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
        inverse=inverse,
    )
