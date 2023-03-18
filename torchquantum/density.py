import imp
import torch
import torch.nn as nn
import numpy as np
import functools
import torchquantum.functional as tqf
import torchquantum.Dfunc as dfunc
import torchquantum as tq
import copy
from torchquantum.states import QuantumState
from torchquantum.macro import C_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from typing import Union, List, Iterable


__all__ = ["DensityMatrix"]


class DensityMatrix(nn.Module):
    def __init__(self, n_wires: int, bsz: int = 1):
        """Init function for DensityMatrix class(Density Operator)
        Args:
            n_wires (int): how many qubits for the densityMatrix.
        """
        super().__init__()

        self.n_wires = n_wires
        """
        For example, when n_wires=3
        matrix[001110] denotes the index of |001><110|=|index1><index2|
        Set Initial value the density matrix of the pure state |00...00>
        """
        _matrix = torch.zeros(2 ** (2 * self.n_wires), dtype=C_DTYPE)
        _matrix[0] = 1 + 0j
        _matrix = torch.reshape(_matrix, [2] * (2 * self.n_wires))
        self.register_buffer("matrix", _matrix)

        repeat_times = [bsz] + [1] * len(self.matrix.shape)
        self._matrix = self.matrix.repeat(*repeat_times)
        self.register_buffer("matrix", self._matrix)

        """
        Whether or not calculate by states
        """
        self._calc_by_states = True

        """
        Remember whether or not a standard matrix on a given wire is contructed
        """
        self.construct = {}
        for key in tqf.func_name_dict.keys():
            self.construct[key] = [False] * n_wires

        """
        Store the constructed operator matrix
        """
        self.operator_matrix = {}
        for key in tqf.func_name_dict.keys():
            self.operator_matrix[key] = {}

        """
        Preserve the probability of all pure states. has the form [(p1,s1),(p2,s2),(p3,s3),...]
              2**n 2**n 2**n
         Matrix  3 purestate
        """
        self.state_list = []
        for i in range(0, bsz):
            self.state_list.append((1, QuantumState(n_wires)))

    def set_calc_by_states(self, val):
        self._calc_by_states = val

    def update_matrix_from_states(self):
        """Update the density matrix value from all pure states"""
        _matrix = torch.zeros(2 ** (2 * self.n_wires), dtype=C_DTYPE)
        _matrix = torch.reshape(_matrix, [2**self.n_wires, 2**self.n_wires])
        self.register_buffer("matrix", _matrix)
        bsz = self.matrix.shape[0]
        repeat_times = [bsz] + [1] * len(self.matrix.shape)
        self._matrix = self.matrix.repeat(*repeat_times)
        for i in range(0, bsz):
            for p, state in self.state_list:
                self._matrix[i] = self._matrix[i] + p * state.density_matrix()[0][:][:]
        self.register_buffer("matrix", self._matrix)

    def vector(self):
        return torch.reshape(_matrix, [2 ** (2 * self.n_wires)])

    def print_2d(self, index):
        """Print the matrix value of matrix[index]"""
        _matrix = torch.reshape(self._matrix[index], [2**self.n_wires] * 2)
        print(_matrix)

    def trace(self, index):
        """Return the trace of the DensityMatrix of matrix[index]"""
        return torch.trace(self._matrix[index])

    def positive_semidefinite(self, index):
        """Check whether the matrix is positive semidefinite by Sylvester's_criterion"""
        return np.all(np.linalg.eigvals(self._matrix[index]) > 0)

    def check_valid(self):
        """Check whether the matrix has trace 1 and is positive semidefinite"""
        for i in range(0, self._matrix.shape[0]):
            if self.trace(i) != 1 or not self.positive_semidefinite(i):
                return False
        return True

    def spectral(self, index):
        """Return the spectral of the DensityMatrix"""
        return list(np.linalg.eigvals(self._matrix[index]))

    def tensor(self, other):
        """Return self tensor other(Notice the order)
        Args:
            other (DensityMatrix: Another density matrix
        """
        self._matrix = torch.kron(self._matrix, other._matrix)

    def expand(self, other):
        """Return other tensor self(Notice the order)
        Args:
            other (DensityMatrix: Another density matrix
        """
        self._matrix = torch.kron(other._matrix, self._matrix)

    def clone_matrix(self, existing_matrix: torch.Tensor):
        self._matrix = existing_matrix.clone()

    def set_matrix(self, matrix: Union[torch.tensor, List]):
        matrix = torch.tensor(matrix, dtype=C_DTYPE).to(self.matrix.device)
        bsz = matrix.shape[0]
        self.matrix = torch.reshape(
            matrix, [bsz] + [2 ** (2 * self.n_wires), 2 ** (2 * self.n_wires)]
        )

    def evolve(self, operator):
        """Evolve the density matrix in batchmode
        operator has size [2**(2*self.n_wires),2**(2*self.n_wires)]
        """
        """Convert the matrix to vector of shape [bsz,2**(2*self.n_wires)]
           Return U rho U^\dagger
        """
        bsz = self.matrix.shape[0]
        expand_shape = [bsz] + list(operator.shape)

        new_matrix = operator.expand(expand_shape).bmm(
            torch.reshape(self.matrix, [bsz, 2 ** (2 * self.n_wires)])
        )
        self.matrix = torch.reshape(
            new_matrix, [bsz, 2**self.n_wires, 2**self.n_wires]
        )

    def expectation(self):
        """Expectation of a measurement"""
        return

    def set_from_state(self, probs, states: Union[torch.Tensor, List]):
        """Get the density matrix of a mixed state.
        Args:
          probs:List of probability of each state
          states:List of state.
        For example:
          probs:[0.5,0.5],states:[|00>,|11>]
        Then the corresponding matrix is: 0.5|00><00|+0.5|11><11|
         0.5, 0, 0, 0
         0  , 0, 0, 0
         0  , 0, 0, 0
         0 ,  0, 0, 0.5
         self._matrix[00][00]=self._matrix[11][11]=0.5
        """
        for i in range(0, len(probs)):
            self.state_list
        _matrix = torch.zeros(2 ** (2 * self.n_wires), dtype=C_DTYPE)
        _matrix = torch.reshape(_matrix, [2**self.n_wires, 2**self.n_wires])
        for i in range(0, len(probs)):
            row = torch.reshape(states[i], [2**self.n_wires, 1])
            col = torch.reshape(states[i], [1, 2**self.n_wires])
            _matrix = _matrix + probs[i] * torch.matmul(row, col)
        self.matrix = torch.reshape(_matrix, [2] * (2 * self.n_wires))
        return

    def _add(self, other):
        """Return self + other
        Args:
            other (complex): a complex number.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        if not self._matrix.shape == other._matrix.shape:
            raise ("Two density matrix must have the same shape.")
        ret = copy.copy(self)
        ret._matrix = self.matrix + other._matrix
        return ret

    def _multiply(self, other):
        """Return other * self.
        Args:
            other (complex): a complex number.
        """
        ret = copy.copy(self)
        ret._matrix = other * self._matrix
        return ret

    def purity(self):
        """Calculate the purity of the DensityMatrix defined as \gamma=tr(\rho^2)"""
        return torch.trace(torch.matmul(self._matrix, self._matrix))

    def partial_trace(self, dims: List[int]):
        """Calculate the partial trace of given sub-dimension, return a new density_matrix
        Args:
            dims:The list of sub-dimension
            For example, If we have 3 qubit, the matrix shape is (8,8),
            We want to do partial trace to qubit 0,2, dims=[0,2].
            First, the matrix should be reshped to (2,2,2,2,2,2)
            then we call  np.einsum('ijiiqi->jq', reshaped_dm)
        """
        return False

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"Density Matrix"

    def hadamard(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.hadamard(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def shadamard(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.shadamard(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def paulix(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.paulix(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def pauliy(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.pauliy(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def pauliz(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.pauliz(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def i(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.i(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def s(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.s(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def t(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.t(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def sx(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.sx(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def cnot(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.cnot(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def cz(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.cz(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def cy(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.cy(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def swap(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.swap(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def sswap(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.sswap(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def cswap(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.cswap(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def toffoli(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):
        dfunc.toffoli(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def multicnot(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        dfunc.multicnot(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def multixcnot(
        self,
        wires: Union[List[int], int],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        dfunc.multixcnot(self, wires=wires, inverse=inverse, comp_method=comp_method)

    def rx(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.rx(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def ry(
        self,
        wires: Union[List[int], int],
        params: torch.Tensor,
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.ry(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def rz(
        self,
        wires: Union[List[int], int],
        params: torch.Tensor,
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.rz(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def rxx(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.rxx(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def ryy(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.ryy(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def rzz(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.rzz(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def rzx(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.rzx(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def phaseshift(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.phaseshift(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def rot(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.rot(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def multirz(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.multirz(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def crx(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.crx(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def cry(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.cry(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def crz(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.crz(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def crot(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.crot(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def u1(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.u1(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def u2(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.u2(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def u3(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.u3(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def cu1(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.cu1(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def cu2(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.cu2(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def cu3(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        dfunc.cu3(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def qubitunitary(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.tensor(params, dtype=C_DTYPE)
        else:
            params = torch.tensor([params], dtype=C_DTYPE)

        dfunc.qubitunitary(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def qubitunitaryfast(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.tensor(params, dtype=C_DTYPE)
        else:
            params = torch.tensor([params], dtype=C_DTYPE)

        dfunc.qubitunitaryfast(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def qubitunitarystrict(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.tensor(params, dtype=C_DTYPE)
        else:
            params = torch.tensor([params], dtype=C_DTYPE)

        dfunc.qubitunitarystrict(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    def single_excitation(
        self,
        wires: Union[List[int], int],
        params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
        inverse: bool = False,
        comp_method: str = "bmm",
    ):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        dfunc.single_excitation(
            self, wires=wires, params=params, inverse=inverse, comp_method=comp_method
        )

    h = hadamard
    sh = shadamard
    x = paulix
    y = pauliy
    z = pauliz
    xx = rxx
    yy = ryy
    zz = rzz
    zx = rzx
    cx = cnot
    ccnot = toffoli
    ccx = toffoli
    u = u3
    cu = cu3
    p = phaseshift
    cp = cu1
    cr = cu1
    cphase = cu
