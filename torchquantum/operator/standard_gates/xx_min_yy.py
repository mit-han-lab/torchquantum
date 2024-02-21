from ..op_types import Observable, Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class XXMINYY(Operation, metaclass=ABCMeta):
    """Class for XXMinusYY gate."""

    num_params = 2
    num_wires = 2
    op_name = "xxminyy"
    func = staticmethod(tqf.xxminyy_matrix)

    @classmethod
    def _matrix(cls, params):
        return tqf.xxminyy_matrix(params)
