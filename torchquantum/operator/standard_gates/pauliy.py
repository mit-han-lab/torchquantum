from ..op_types import Observable, Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class PauliY(Observable, metaclass=ABCMeta):
    """Class for Pauli Y Gate."""

    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    op_name = "pauliy"
    matrix = mat_dict["pauliy"]
    func = staticmethod(tqf.pauliy)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return [tq.PauliZ(), tq.S(), tq.Hadamard()]


class CY(Operation, metaclass=ABCMeta):
    """Class for CY Gate."""

    num_params = 0
    num_wires = 2
    op_name = "cy"
    matrix = mat_dict["cy"]
    func = staticmethod(tqf.cy)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix
