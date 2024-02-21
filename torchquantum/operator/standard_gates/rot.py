from ..op_types import Observable, Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class Rot(Operation, metaclass=ABCMeta):
    """Class for Rotation Gate."""

    num_params = 3
    num_wires = 1
    op_name = "rot"
    func = staticmethod(tqf.rot)

    @classmethod
    def _matrix(cls, params):
        return tqf.rot_matrix(params)


class CRot(Operation, metaclass=ABCMeta):
    """Class for Controlled Rotation gate."""

    num_params = 3
    num_wires = 2
    op_name = "crot"
    func = staticmethod(tqf.crot)

    @classmethod
    def _matrix(cls, params):
        return tqf.crot_matrix(params)
