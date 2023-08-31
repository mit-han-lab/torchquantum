from .op_types import Operation
from abc import ABCMeta
from ..macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional.functionals as tqf


class SX(Operation, metaclass=ABCMeta):
    """Class for SX Gate."""

    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1j], dtype=C_DTYPE)
    matrix = mat_dict["sx"]
    func = staticmethod(tqf.sx)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class CSX(Operation, metaclass=ABCMeta):
    """Class for CSX Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["csx"]
    func = staticmethod(tqf.csx)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class C3SX(Operation, metaclass=ABCMeta):
    """Class for C3SX Gate."""

    num_params = 0
    num_wires = 4
    matrix = mat_dict["c3sx"]
    func = staticmethod(tqf.c3sx)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class SXDG(Operation, metaclass=ABCMeta):
    """Class for SXDG Gate."""

    num_params = 0
    num_wires = 1
    matrix = mat_dict["sxdg"]
    func = staticmethod(tqf.sxdg)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix
