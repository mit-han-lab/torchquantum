from ..op_types import Observable, DiagonalOperation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class PauliZ(Observable, metaclass=ABCMeta):
    """Class for Pauli Z Gate."""

    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    op_name = "pauliz"
    matrix = mat_dict["pauliz"]
    func = staticmethod(tqf.pauliz)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return []


class CZ(DiagonalOperation, metaclass=ABCMeta):
    """Class for CZ Gate."""

    num_params = 0
    num_wires = 2
    eigvals = torch.tensor([1, 1, 1, -1], dtype=C_DTYPE)
    op_name = "cz"
    matrix = mat_dict["cz"]
    func = staticmethod(tqf.cz)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class CCZ(DiagonalOperation, metaclass=ABCMeta):
    """Class for CCZ Gate."""

    num_params = 0
    num_wires = 3
    op_name = "ccz"
    matrix = mat_dict["ccz"]
    eigvals = torch.tensor([1, 1, 1, 1, 1, 1, 1, -1], dtype=C_DTYPE)
    func = staticmethod(tqf.ccz)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals
