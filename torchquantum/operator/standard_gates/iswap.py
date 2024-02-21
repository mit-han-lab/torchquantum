from ..op_types import Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class ISWAP(Operation, metaclass=ABCMeta):
    """Class for ISWAP Gate."""

    num_params = 0
    num_wires = 2
    op_name = "iswap"
    matrix = mat_dict["iswap"]
    func = staticmethod(tqf.iswap)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix
