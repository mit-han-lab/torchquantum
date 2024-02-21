from ..op_types import Observable, AnyWires
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf

class QFT(Observable, metaclass=ABCMeta):
    """Class for Quantum Fourier Transform."""

    num_params = 0
    num_wires = AnyWires
    op_name = "qft"
    func = staticmethod(tqf.qft)

    @classmethod
    def _matrix(cls, params, n_wires):
        return tqf.qft_matrix(n_wires)
