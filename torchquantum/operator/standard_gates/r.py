from ..op_types import DiagonalOperation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class R(DiagonalOperation, metaclass=ABCMeta):
    """Class for R Gate."""

    num_params = 2
    num_wires = 1
    op_name = "r"
    func = staticmethod(tqf.r)

    @classmethod
    def _matrix(cls, params):
        return tqf.r_matrix(params)
