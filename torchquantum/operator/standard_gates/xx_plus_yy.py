from ..op_types import Observable, Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class XXPLUSYY(Operation, metaclass=ABCMeta):
    """Class for XXPlusYY gate."""

    num_params = 2
    num_wires = 2
    op_name = "xxplusyy"
    func = staticmethod(tqf.xxplusyy_matrix)

    @classmethod
    def _matrix(cls, params):
        return tqf.xxplusyy_matrix(params)
