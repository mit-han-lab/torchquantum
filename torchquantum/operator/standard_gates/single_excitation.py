from ..op_types import Operator
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class SingleExcitation(Operator, metaclass=ABCMeta):
    """Class for SingleExcitation gate."""

    num_params = 1
    num_wires = 2
    op_name = "singleexcitation"
    func = staticmethod(tqf.singleexcitation)

    @classmethod
    def _matrix(cls, params):
        return tqf.singleexcitation_matrix(params)
