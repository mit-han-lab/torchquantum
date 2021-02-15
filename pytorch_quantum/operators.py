import torch
import torch.nn as nn
import pytorch_quantum as tq
import pytorch_quantum.functional as tqf
import numpy as np
import logging

from abc import ABCMeta
from .macro import C_DTYPE, F_DTYPE, INV_SQRT2

logger = logging.getLogger()


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
        try:
            assert self.name in self.fixed_ops or \
                   self.has_params ^ (params is not None)
        except AssertionError as err:
            logger.exception(f"Parameterized gate either has its "
                             f"own parameters or has input as parameters")
            raise err

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
    def __init__(self,
                 has_params: bool = False,
                 trainable: bool = False,
                 init_params=None):
        super().__init__()

        try:
            assert not (trainable and not has_params)
        except AssertionError:
            has_params = True
            logger.warning(f"Module must have parameters to be trainable; "
                           f"Switched 'has_params' to True.")

        self.has_params = has_params
        self.trainable = trainable
        if self.has_params:
            self.params = self.build_params(trainable=self.trainable)
            self.reset_params(init_params)

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


class DiagonalOperation(Operation):
    @classmethod
    def _eigvals(cls, params):
        raise NotImplementedError

    @property
    def eigvals(self):
        return super().eigvals

    @classmethod
    def _matrix(cls, params):
        return torch.diag(cls._eigvals(params))



class Hadamard(Observable, Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = torch.tensor([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]],
                          dtype=C_DTYPE)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return [tq.RY(has_params=True,
                      trainable=False,
                      init_params=-np.pi / 4)]


class PauliX(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = torch.tensor([[0, 1], [1, 0]], dtype=C_DTYPE)
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
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = torch.tensor([[0, -1j], [1j, 0]], dtype=C_DTYPE)
    func = staticmethod(tqf.pauliy)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return [tq.PauliZ(), tq.S(), tq.Hadamard()]


class PauliZ(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = torch.tensor([[1, 0], [0, -1]], dtype=C_DTYPE)
    func = staticmethod(tqf.pauliz)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return []


class S(DiagonalOperation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1j], dtype=C_DTYPE)
    matrix = torch.tensor([[1, 0], [0, 1j]], dtype=C_DTYPE)
    func = staticmethod(tqf.s)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class T(DiagonalOperation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1j], dtype=C_DTYPE)
    matrix = torch.tensor([[1, 0], [0, np.exp(1j * np.pi / 4)]],
                          dtype=C_DTYPE)
    func = staticmethod(tqf.s)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class RX(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.rx)

    def __init__(self,
                 has_params: bool = False,
                 trainable: bool = False,
                 init_params=None):
        super().__init__(
            has_params=has_params,
            trainable=trainable,
            init_params=init_params
        )

    @classmethod
    def _matrix(cls, params):
        return tqf.rx_matrix(params)

    def build_params(self, trainable):
        parameters = nn.Parameter(torch.empty([1, self.num_params],
                                              dtype=F_DTYPE))
        parameters.requires_grad = True if trainable else False
        self.register_parameter('rx_theta', parameters)
        return parameters

    def reset_params(self, init_params=None):
        if init_params is not None:
            torch.nn.init.constant_(self.params, init_params)
        else:
            torch.nn.init.uniform_(self.params, 0, 2 * np.pi)


class RY(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.ry)

    def __init__(self,
                 has_params: bool = False,
                 trainable: bool = False,
                 init_params=None):
        super().__init__(
            has_params=has_params,
            trainable=trainable,
            init_params=init_params
        )

    @classmethod
    def _matrix(cls, params):
        return tqf.ry_matrix(params)

    def build_params(self, trainable):
        parameters = nn.Parameter(torch.empty([1, self.num_params],
                                              dtype=F_DTYPE))
        parameters.requires_grad = True if trainable else False
        self.register_parameter('ry_theta', parameters)
        return parameters

    def reset_params(self, init_params=None):
        if init_params is not None:
            torch.nn.init.constant_(self.params, init_params)
        else:
            torch.nn.init.uniform_(self.params, 0, 2 * np.pi)


class RZ(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.rz)

    def __init__(self,
                 has_params: bool = False,
                 trainable: bool = False,
                 init_params=None):
        super().__init__(
            has_params=has_params,
            trainable=trainable,
            init_params=init_params
        )

    @classmethod
    def _matrix(cls, params):
        return tqf.rz_matrix(params)

    def build_params(self, trainable):
        parameters = nn.Parameter(torch.empty([1, self.num_params],
                                              dtype=F_DTYPE))
        parameters.requires_grad = True if trainable else False
        self.register_parameter('rz_theta', parameters)
        return parameters

    def reset_params(self, init_params=None):
        if init_params is not None:
            torch.nn.init.constant_(self.params, init_params)
        else:
            torch.nn.init.uniform_(self.params, 0, 2 * np.pi)
