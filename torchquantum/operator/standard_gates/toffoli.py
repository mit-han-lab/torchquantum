from ..op_types import Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class Toffoli(Operation, metaclass=ABCMeta):
    """Class for Toffoli Gate."""

    num_params = 0
    num_wires = 3
    op_name = "toffoli"
    matrix = mat_dict["toffoli"]
    func = staticmethod(tqf.toffoli)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


CCX = Toffoli


class RC3X(Operation, metaclass=ABCMeta):
    """Class for RC3X Gate."""

    num_params = 0
    num_wires = 4
    op_name = "rc3x"
    matrix = mat_dict["rc3x"]
    func = staticmethod(tqf.rc3x)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class RCCX(Operation, metaclass=ABCMeta):
    """Class for RCCX Gate."""

    num_params = 0
    num_wires = 3
    op_name = "rccx"
    matrix = mat_dict["rccx"]
    func = staticmethod(tqf.rccx)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix
