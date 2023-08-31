from .op_types import Operator
from abc import ABCMeta
from ..macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional.functionals as tqf


class SingleExcitation(Operator, metaclass=ABCMeta):
    """Class for SingleExcitation gate."""

    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.singleexcitation)

    @classmethod
    def _matrix(cls, params):
        return tqf.singleexcitation_matrix(params)
