import functools
import math
import torch
import torch.nn as nn
import pytorch_quantum as tq
import numpy as np

from string import ascii_lowercase as ABC

from abc import ABCMeta

ABC_ARRAY = np.array(list(ABC))
INV_SQRT2 = 1 / math.sqrt(2)
C_DTYPE = torch.complex64


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

    def _apply_unitary_einsum(self, state, mat, wires):
        device_wires = wires

        total_wires = len(state.shape) - 1

        if len(mat.shape) > 2:
            is_batch_unitary = True
            bsz = mat.shape[0]
            shape_extension = [bsz]
            assert state.shape[0] == bsz
        else:
            is_batch_unitary = False
            shape_extension = []

        mat = torch.reshape(mat, shape_extension + [2] * self.num_wires * 2)

        mat = mat.type(C_DTYPE).to(state)

        # Tensor indices of the quantum state
        state_indices = ABC[: total_wires]

        # Indices of the quantum state affected by this operation
        affected_indices = "".join(ABC_ARRAY[list(device_wires)].tolist())

        # All affected indices will be summed over, so we need the same number
        # of new indices
        new_indices = ABC[total_wires: total_wires + len(device_wires)]

        # The new indices of the state are given by the old ones with the
        # affected indices replaced by the new_indices
        new_state_indices = functools.reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0],
                                                            idx_pair[1]),
            zip(affected_indices, new_indices),
            state_indices,
        )

        # cannot support too many qubits...
        assert ABC[-1] not in state_indices + new_state_indices + new_indices \
               + affected_indices

        state_indices = ABC[-1] + state_indices
        new_state_indices = ABC[-1] + new_state_indices
        if is_batch_unitary:
            new_indices = ABC[-1] + new_indices

        # We now put together the indices in the notation numpy einsum
        # requires
        einsum_indices = f"{new_indices}{affected_indices}," \
                         f"{state_indices}->{new_state_indices}"

        new_state = torch.einsum(einsum_indices, mat, state)

        return new_state

    def forward(self, q_device: tq.QuantumDevice, wires, params=None):
        # assert type(self) in self.fixed_ops or \
        #        self.trainable ^ (params is not None), \
        #        f"Parameterized gate either has its own parameters or " \
        #        f"has input as parameters"

        if params is not None:
            params = params.unsqueeze(-1) if params.dim() == 1 else params
            self.params = params


        state = q_device.states
        matrix = self._get_unitary_matrix()
        wires = [wires] if isinstance(wires, int) else wires

        q_device.states = self._apply_unitary_einsum(state, matrix, wires)


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
            self.params = self.init_params()

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

    def __init__(self, trainable: bool = False):
        super().__init__(trainable=trainable)

    @classmethod
    def _matrix(cls, params):
        theta = params.type(C_DTYPE)
        # theta = params
        """
        Seems to be a pytorch bug. Have to explicitly cast the theta to a 
        complex number. If directly theta = params, then get error:
        
        allow_unreachable=True, accumulate_grad=True)  # allow_unreachable flag
        RuntimeError: Expected isFloatingType(grad.scalar_type()) || 
        (input_is_complex == grad_is_complex) to be true, but got false.  
        (Could this error message be improved?  
        If so, please report an enhancement request to PyTorch.)
        
        """

        c = torch.cos(theta / 2)
        js = 1j * torch.sin(theta / 2)

        return torch.stack([torch.cat([c, js], dim=-1),
                            torch.cat([js, c], dim=-1)], dim=-1).squeeze(0)

    def init_params(self):
        parameter = nn.Parameter(2 * np.pi * torch.randn(
            [1, self.num_params]))
        self.register_parameter('rx_theta', parameter)

        return parameter


h = Hadamard
x = PauliX
y = PauliY
z = PauliZ
rx = RX
