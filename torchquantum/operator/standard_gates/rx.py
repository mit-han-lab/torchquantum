from ..op_types import Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class RX(Operation, metaclass=ABCMeta):
    """Class for RX Gate."""

    num_params = 1
    num_wires = 1
    op_name = "rx"
    func = staticmethod(tqf.rx)

    @classmethod
    def _matrix(cls, params):
        return tqf.rx_matrix(params)


class RXX(Operation, metaclass=ABCMeta):
    """Class for RXX Gate."""

    num_params = 1
    num_wires = 2
    op_name = "rxx"
    func = staticmethod(tqf.rxx)

    @classmethod
    def _matrix(cls, params):
        return tqf.rxx_matrix(params)


class CRX(Operation, metaclass=ABCMeta):
    """Class for Controlled Rotation X gate."""

    num_params = 1
    num_wires = 2
    op_name = "crx"
    func = staticmethod(tqf.crx)

    @classmethod
    def _matrix(cls, params):
        return tqf.crx_matrix(params)
