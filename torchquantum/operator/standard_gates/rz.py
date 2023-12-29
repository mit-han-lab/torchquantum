from ..op_types import *
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class RZ(DiagonalOperation, metaclass=ABCMeta):
    """Class for RZ Gate."""

    num_params = 1
    num_wires = 1
    op_name = "rz"
    func = staticmethod(tqf.rz)

    @classmethod
    def _matrix(cls, params):
        return tqf.rz_matrix(params)


class MultiRZ(DiagonalOperation, metaclass=ABCMeta):
    """Class for Multi-qubit RZ Gate."""

    num_params = 1
    num_wires = AnyWires
    op_name = "multirz"
    func = staticmethod(tqf.multirz)

    @classmethod
    def _matrix(cls, params, n_wires):
        return tqf.multirz_matrix(params, n_wires)


class RZZ(DiagonalOperation, metaclass=ABCMeta):
    """Class for RZZ Gate."""

    num_params = 1
    num_wires = 2
    op_name = "rzz"
    func = staticmethod(tqf.rzz)

    @classmethod
    def _matrix(cls, params):
        return tqf.rzz_matrix(params)


class RZX(Operation, metaclass=ABCMeta):
    """Class for RZX Gate."""

    num_params = 1
    num_wires = 2
    op_name = "rzx"
    func = staticmethod(tqf.rzx)

    @classmethod
    def _matrix(cls, params):
        return tqf.rzx_matrix(params)


class CRZ(Operation, metaclass=ABCMeta):
    """Class for Controlled Rotation Z gate."""

    num_params = 1
    num_wires = 2
    op_name = "crz"
    func = staticmethod(tqf.crz)

    @classmethod
    def _matrix(cls, params):
        return tqf.crz_matrix(params)
