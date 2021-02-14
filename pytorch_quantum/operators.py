import functools
import math
import torch
import torch.nn as nn
import pytorch_quantum as tq
import numpy as np

from abc import ABCMeta
from .macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from .functional import rx, rx_matrix


class Operator(nn.Module):
    # fixed_ops = [
    #     tq.Hadamard,
    #     tq.PauliX,
    #     tq.PauliY,
    #     tq.PauliZ
    # ]
    # parameterized_ops = [
    #     tq.RX
    # ]
    def __init__(self):
        super().__init__()
        self.params = None

    @classmethod
    def _matrix(cls, params):
        raise NotImplementedError

    @property
    def matrix(self):
        return self._matrix(self.params)

    @classmethod
    def _eigvals(cls, params):
        raise NotImplementedError

    @property
    def eigvals(self):
        return self._eigvals(self.params)

    def _get_unitary_matrix(self):
        return self.matrix

    def forward(self, q_device: tq.QuantumDevice, wires, params=None):
        # assert type(self) in self.fixed_ops or \
        #        self.trainable ^ (params is not None), \
        #        f"Parameterized gate either has its own parameters or " \
        #        f"has input as parameters"

        if params is not None:
            self.params = params

        self.func(q_device, wires, self.params)


class Observable(Operator, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.return_type = None

    def diagonalizing_gates(self):
        raise NotImplementedError


class Operation(Operator, metaclass=ABCMeta):
    def __init__(self, trainable: bool = False):
        super().__init__()
        self.trainable = trainable
        if self.trainable:
            self.params = self.build_params()
            self.reset_params()

    @property
    def matrix(self):
        op_matrix = self._matrix(self.params)

        return op_matrix

    @property
    def eigvals(self):
        op_eigvals = self._eigvals(self.params)

        return op_eigvals

    def init_params(self):
        raise NotImplementedError


class Hadamard(Observable, Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1])
    matrix = torch.tensor([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]])

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        # FIXME
        return []


class PauliX(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1])
    matrix = torch.tensor([[0., 1.], [1., 0.]])

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return [tq.Hadamard()]


class PauliY(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1])
    matrix = torch.tensor([[0., -1j], [1j, 0.]])

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        # FIXME
        return []


class PauliZ(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1])
    matrix = torch.tensor([[1., 0.], [0., -1.]])

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return []


class RX(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 1
    func = staticmethod(rx)

    def __init__(self, trainable: bool = False):
        super().__init__(trainable=trainable)

    @classmethod
    def _matrix(cls, params):
        return rx_matrix(params)

    def build_params(self):
        parameters = nn.Parameter(torch.empty([1, self.num_params], dtype=
                                              F_DTYPE))
        self.register_parameter('rx_theta', parameters)
        return parameters

    def reset_params(self):
        torch.nn.init.uniform_(self.params, 0, 2 * np.pi)


class RY(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 1

    def __init__(self, trainable: bool = False):
        super().__init__(trainable=trainable)

    @classmethod
    def _matrix(cls, params):
        theta = params.type(C_DTYPE)

        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)

        return torch.stack([torch.cat([c, -s], dim=-1),
                            torch.cat([s, c], dim=-1)], dim=-1).squeeze(0)

    def build_params(self):
        parameters = nn.Parameter(torch.empty([1, self.num_params],
                                              dtype=F_DTYPE))
        self.register_parameter('ry_theta', parameters)
        return parameters

    def reset_params(self):
        torch.nn.init.uniform_(self.params, 0, 2 * np.pi)


class RZ(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 1

    def __init__(self, trainable: bool = False):
        super().__init__(trainable=trainable)

    @classmethod
    def _matrix(cls, params):
        theta = params.type(C_DTYPE)

        c = torch.cos(theta / 2)
        js = 1j * torch.sin(theta / 2)

        return torch.stack([torch.cat([c + js, 0], dim=-1),
                            torch.cat([0, c - js], dim=-1)], dim=-1).squeeze(0)

    def build_params(self):
        parameters = nn.Parameter(torch.empty([1, self.num_params],
                                              dtype=F_DTYPE))
        self.register_parameter('rz_theta', parameters)
        return parameters

    def reset_params(self):
        torch.nn.init.uniform_(self.params, 0, 2 * np.pi)

