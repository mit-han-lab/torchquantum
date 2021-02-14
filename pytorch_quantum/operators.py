import functools
import math
import torch
import torch.nn as nn
import pytorch_quantum as tq
import pytorch_quantum.functional as tqf
import numpy as np

from abc import ABCMeta
from .macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY, INV_SQRT2


class Operator(nn.Module):
    fixed_ops = [
        'Hadamard',
        'PauliX',
        'PauliY',
        'PauliZ'
    ]
    parameterized_ops = [
        'RX',
        'RY',
        'RZ',
    ]

    @property
    def name(self):
        """String for the name of the operator."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def __init__(self):
        super().__init__()
        self.params = None
        self._name = self.__class__.__name__

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
        assert self.name in self.fixed_ops or \
               self.trainable ^ (params is not None), \
               f"Parameterized gate either has its own parameters or " \
               f"has input as parameters"

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
    func = staticmethod(tqf.paulix)

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
    func = staticmethod(tqf.pauliy)

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
    func = staticmethod(tqf.pauliz)

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
    func = staticmethod(tqf.rx)

    def __init__(self, trainable: bool = False):
        super().__init__(trainable=trainable)

    @classmethod
    def _matrix(cls, params):
        return tqf.rx_matrix(params)

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
    func = staticmethod(tqf.ry)

    def __init__(self, trainable: bool = False):
        super().__init__(trainable=trainable)

    @classmethod
    def _matrix(cls, params):
        return tqf.ry_matrix(params)

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
    func = staticmethod(tqf.rz)

    def __init__(self, trainable: bool = False):
        super().__init__(trainable=trainable)

    @classmethod
    def _matrix(cls, params):
        return tqf.rz_matrix(params)

    def build_params(self):
        parameters = nn.Parameter(torch.empty([1, self.num_params],
                                              dtype=F_DTYPE))
        self.register_parameter('rz_theta', parameters)
        return parameters

    def reset_params(self):
        torch.nn.init.uniform_(self.params, 0, 2 * np.pi)

