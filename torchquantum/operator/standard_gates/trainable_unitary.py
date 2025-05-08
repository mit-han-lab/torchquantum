from ..op_types import *
from abc import ABCMeta
from torchquantum.macro import C_DTYPE
import torchquantum as tq
import torch
import torch.nn as nn
from torchquantum.functional import mat_dict
import torchquantum.functional as tqf


class TrainableUnitary(Operation, metaclass=ABCMeta):
    """Class for TrainableUnitary Gate."""

    num_params = AnyNParams
    num_wires = AnyWires
    op_name = "trainableunitary"
    func = staticmethod(tqf.qubitunitaryfast)

    def build_params(self, trainable):
        """Build the parameters for the gate.

        Args:
            trainable (bool): Whether the parameters are trainble.

        Returns:
            torch.Tensor: Parameters.

        """
        parameters = nn.Parameter(
            torch.empty(1, 2**self.n_wires, 2**self.n_wires, dtype=C_DTYPE)
        )
        parameters.requires_grad = True if trainable else False
        # self.register_parameter(f"{self.name}_params", parameters)
        return parameters

    def reset_params(self, init_params=None):
        """Reset the parameters.

        Args:
            init_params (torch.Tensor, optional): Initial parameters.

        Returns:
            None.

        """
        mat = torch.randn((1, 2**self.n_wires, 2**self.n_wires), dtype=C_DTYPE)
        U, Sigma, V = torch.svd(mat)
        self.params.data.copy_(U.matmul(V.permute(0, 2, 1)))

    @staticmethod
    def _matrix(self, params):
        return tqf.qubitunitaryfast(params)


class TrainableUnitaryStrict(TrainableUnitary, metaclass=ABCMeta):
    """Class for Strict Unitary matrix gate."""

    num_params = AnyNParams
    num_wires = AnyWires
    op_name = "trainableunitarystrict"
    func = staticmethod(tqf.qubitunitarystrict)
