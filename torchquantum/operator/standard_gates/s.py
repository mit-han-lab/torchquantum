from ..op_types import Operation, DiagonalOperation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class S(DiagonalOperation, metaclass=ABCMeta):
    """Class for S Gate."""

    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1j], dtype=C_DTYPE)
    op_name = "s"
    matrix = mat_dict["s"]
    func = staticmethod(tqf.s)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class SDG(Operation, metaclass=ABCMeta):
    """Class for SDG Gate."""

    num_params = 0
    num_wires = 1

    op_name = "sdg"
    matrix = mat_dict["sdg"]
    func = staticmethod(tqf.sdg)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class CS(Operation, metaclass=ABCMeta):
    """Class for CS Gate."""

    num_params = 0
    num_wires = 2
    op_name = "cs"
    matrix = mat_dict["cs"]
    eigvals = torch.tensor([1, 1, 1, 1j], dtype=C_DTYPE)
    func = staticmethod(tqf.cs)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class CSDG(DiagonalOperation, metaclass=ABCMeta):
    """Class for CS Dagger Gate."""

    num_params = 0
    num_wires = 2
    op_name = "csdg"
    matrix = mat_dict["csdg"]
    eigvals = torch.tensor([1, 1, 1, -1j], dtype=C_DTYPE)
    func = staticmethod(tqf.csdg)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals
