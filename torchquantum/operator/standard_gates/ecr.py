from ..op_types import Operation
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class ECR(Operation, metaclass=ABCMeta):
    """Class for Echoed Cross Resonance Gate."""

    num_params = 0
    num_wires = 2
    op_name = "ecr"
    matrix = mat_dict["ecr"]
    func = staticmethod(tqf.ecr)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


EchoedCrossResonance = ECR
EchoedCrossResonance.name = "echoedcrossresonance"
