"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional.functionals as tqf
import numpy as np

from enum import IntEnum
from torchquantum.functional import mat_dict
from torchquantum.util.quantization.clifford_quantization import CliffordQuantizer
from abc import ABCMeta
from ..macro import C_DTYPE, F_DTYPE
from torchpack.utils.logging import logger
from typing import Iterable, Union, List

from .op_types import *
from .hadamard import *
from .paulix import *
from .pauliy import *
from .pauliz import *
from .i import *
from .s import *
from .t import *
from .sx import *

__all__ = [
    "op_name_dict",
    "Operator",
    "Operation",
    "DiagonalOperation",
    "Observable",
    "Hadamard",
    "H",
    "SHadamard",
    "SH",
    "PauliX",
    "PauliY",
    "PauliZ",
    "I",
    "S",
    "T",
    "SX",
    "CNOT",
    "CZ",
    "CY",
    "RX",
    "RY",
    "RZ",
    "RXX",
    "RYY",
    "RZZ",
    "RZX",
    "SWAP",
    "SSWAP",
    "CSWAP",
    "Toffoli",
    "PhaseShift",
    "Rot",
    "MultiRZ",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "U1",
    "U2",
    "U3",
    "CU",
    "CU1",
    "CU2",
    "CU3",
    "QubitUnitary",
    "QubitUnitaryFast",
    "TrainableUnitary",
    "TrainableUnitaryStrict",
    "MultiCNOT",
    "MultiXCNOT",
    "Reset",
    "SingleExcitation",
    "EchoedCrossResonance",
    "ECR",
    "QFT",
    "SDG",
    "TDG",
    "SXDG",
    "CH",
    "CCZ",
    "ISWAP",
    "CS",
    "CSDG",
    "CSX",
    "CHadamard",
    "CCZ",
    "DCX",
    "XXMINYY",
    "XXPLUSYY",
    "C3X",
    "R",
    "C4X",
    "RC3X",
    "RCCX",
    "GlobalPhase",
    "C3SX",
]


class WiresEnum(IntEnum):
    """Integer enumeration class
    to represent the number of wires
    an operation acts on."""

    AnyWires = -1
    AllWires = 0


class NParamsEnum(IntEnum):
    """Integer enumeration class
    to represent the number of wires
    an operation acts on"""

    AnyNParams = -1


AnyNParams = NParamsEnum.AnyNParams


AllWires = WiresEnum.AllWires
"""IntEnum: An enumeration which represents all wires in the
subsystem. It is equivalent to an integer with value 0."""

AnyWires = WiresEnum.AnyWires
"""IntEnum: An enumeration which represents any wires in the
subsystem. It is equivalent to an integer with value -1."""


class CNOT(Operation, metaclass=ABCMeta):
    """Class for CNOT Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["cnot"]
    func = staticmethod(tqf.cnot)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class CZ(DiagonalOperation, metaclass=ABCMeta):
    """Class for CZ Gate."""

    num_params = 0
    num_wires = 2
    eigvals = np.array([1, 1, 1, -1])
    matrix = mat_dict["cz"]
    func = staticmethod(tqf.cz)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class CY(Operation, metaclass=ABCMeta):
    """Class for CY Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["cy"]
    func = staticmethod(tqf.cy)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class SWAP(Operation, metaclass=ABCMeta):
    """Class for SWAP Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["swap"]
    func = staticmethod(tqf.swap)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class SSWAP(Operation, metaclass=ABCMeta):
    """Class for SSWAP Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["sswap"]
    func = staticmethod(tqf.sswap)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class CSWAP(Operation, metaclass=ABCMeta):
    """Class for CSWAP Gate."""

    num_params = 0
    num_wires = 3
    matrix = mat_dict["cswap"]
    func = staticmethod(tqf.cswap)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class Toffoli(Operation, metaclass=ABCMeta):
    """Class for Toffoli Gate."""

    num_params = 0
    num_wires = 3
    matrix = mat_dict["toffoli"]
    func = staticmethod(tqf.toffoli)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class RX(Operation, metaclass=ABCMeta):
    """Class for RX Gate."""

    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.rx)

    @classmethod
    def _matrix(cls, params):
        return tqf.rx_matrix(params)


class RY(Operation, metaclass=ABCMeta):
    """Class for RY Gate."""

    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.ry)

    @classmethod
    def _matrix(cls, params):
        return tqf.ry_matrix(params)


class RZ(DiagonalOperation, metaclass=ABCMeta):
    """Class for RZ Gate."""

    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.rz)

    @classmethod
    def _matrix(cls, params):
        return tqf.rz_matrix(params)


class PhaseShift(DiagonalOperation, metaclass=ABCMeta):
    """Class for PhaseShift Gate."""

    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.phaseshift)

    @classmethod
    def _matrix(cls, params):
        return tqf.phaseshift_matrix(params)


class Rot(Operation, metaclass=ABCMeta):
    """Class for Rotation Gate."""

    num_params = 3
    num_wires = 1
    func = staticmethod(tqf.rot)

    @classmethod
    def _matrix(cls, params):
        return tqf.rot_matrix(params)


class MultiRZ(DiagonalOperation, metaclass=ABCMeta):
    """Class for Multi-qubit RZ Gate."""

    num_params = 1
    num_wires = AnyWires
    func = staticmethod(tqf.multirz)

    @classmethod
    def _matrix(cls, params, n_wires):
        return tqf.multirz_matrix(params, n_wires)


class RXX(Operation, metaclass=ABCMeta):
    """Class for RXX Gate."""

    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.rxx)

    @classmethod
    def _matrix(cls, params):
        return tqf.rxx_matrix(params)


class RYY(Operation, metaclass=ABCMeta):
    """Class for RYY Gate."""

    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.ryy)

    @classmethod
    def _matrix(cls, params):
        return tqf.ryy_matrix(params)


class RZZ(DiagonalOperation, metaclass=ABCMeta):
    """Class for RZZ Gate."""

    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.rzz)

    @classmethod
    def _matrix(cls, params):
        return tqf.rzz_matrix(params)


class RZX(Operation, metaclass=ABCMeta):
    """Class for RZX Gate."""

    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.rzx)

    @classmethod
    def _matrix(cls, params):
        return tqf.rzx_matrix(params)


class TrainableUnitary(Operation, metaclass=ABCMeta):
    """Class for TrainableUnitary Gate."""

    num_params = AnyNParams
    num_wires = AnyWires
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
    func = staticmethod(tqf.qubitunitarystrict)


class CRX(Operation, metaclass=ABCMeta):
    """Class for Controlled Rotation X gate."""

    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.crx)

    @classmethod
    def _matrix(cls, params):
        return tqf.crx_matrix(params)


class CRY(Operation, metaclass=ABCMeta):
    """Class for Controlled Rotation Y gate."""

    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.cry)

    @classmethod
    def _matrix(cls, params):
        return tqf.cry_matrix(params)


class CRZ(Operation, metaclass=ABCMeta):
    """Class for Controlled Rotation Z gate."""

    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.crz)

    @classmethod
    def _matrix(cls, params):
        return tqf.crz_matrix(params)


class CRot(Operation, metaclass=ABCMeta):
    """Class for Controlled Rotation gate."""

    num_params = 3
    num_wires = 2
    func = staticmethod(tqf.crot)

    @classmethod
    def _matrix(cls, params):
        return tqf.crot_matrix(params)


class U1(DiagonalOperation, metaclass=ABCMeta):
    """Class for Controlled Rotation Y gate.  U1 is the same
    as phaseshift.
    """

    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.u1)

    @classmethod
    def _matrix(cls, params):
        return tqf.u1_matrix(params)


class CU(Operation, metaclass=ABCMeta):
    """Class for Controlled U gate (4-parameter two-qubit gate)."""

    num_params = 4
    num_wires = 2
    func = staticmethod(tqf.cu)

    @classmethod
    def _matrix(cls, params):
        return tqf.cu_matrix(params)


class CU1(DiagonalOperation, metaclass=ABCMeta):
    """Class for controlled U1 gate."""

    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.cu1)

    @classmethod
    def _matrix(cls, params):
        return tqf.cu1_matrix(params)


class U2(Operation, metaclass=ABCMeta):
    """Class for U2 gate."""

    num_params = 2
    num_wires = 1
    func = staticmethod(tqf.u2)

    @classmethod
    def _matrix(cls, params):
        return tqf.u2_matrix(params)


class CU2(Operation, metaclass=ABCMeta):
    """Class for controlled U2 gate."""

    num_params = 2
    num_wires = 2
    func = staticmethod(tqf.cu2)

    @classmethod
    def _matrix(cls, params):
        return tqf.cu2_matrix(params)


class U3(Operation, metaclass=ABCMeta):
    """Class for U3 gate."""

    num_params = 3
    num_wires = 1
    func = staticmethod(tqf.u3)

    @classmethod
    def _matrix(cls, params):
        return tqf.u3_matrix(params)


class CU3(Operation, metaclass=ABCMeta):
    """Class for Controlled U3 gate."""

    num_params = 3
    num_wires = 2
    func = staticmethod(tqf.cu3)

    @classmethod
    def _matrix(cls, params):
        return tqf.cu3_matrix(params)


class QubitUnitary(Operation, metaclass=ABCMeta):
    """Class for controlled Qubit Unitary gate."""

    num_params = AnyNParams
    num_wires = AnyWires
    func = staticmethod(tqf.qubitunitary)

    @classmethod
    def _matrix(cls, params):
        return tqf.qubitunitary_matrix(params)

    def build_params(self, trainable):
        return None

    def reset_params(self, init_params=None):
        self.params = torch.tensor(init_params, dtype=C_DTYPE)
        self.register_buffer(f"{self.name}_unitary", self.params)


class QubitUnitaryFast(Operation, metaclass=ABCMeta):
    """Class for fast implementation of
    controlled Qubit Unitary gate."""

    num_params = AnyNParams
    num_wires = AnyWires
    func = staticmethod(tqf.qubitunitaryfast)

    def __init__(
        self,
        has_params: bool = False,
        trainable: bool = False,
        init_params=None,
        n_wires=None,
        wires=None,
    ):
        super().__init__(
            has_params=True,
            trainable=trainable,
            init_params=init_params,
            n_wires=n_wires,
            wires=wires,
        )

    @classmethod
    def from_controlled_operation(
        cls,
        op,
        c_wires,
        t_wires,
        trainable,
    ):
        """

        Args:
            op: the operation
            c_wires: controlled wires, will only be a set such as 1, [2,3]
            t_wires: can be a list of list of wires, multiple sets
            [[1,2], [3,4]]
            trainable:
        """
        op = op
        c_wires = np.array(c_wires)
        t_wires = np.array(t_wires)
        trainable = trainable
        # self.n_t_wires = op.n_wires
        # assert len(t_wires) == op.n_wires

        orig_u = op.matrix
        orig_u_n_wires = op.n_wires

        wires = []

        if c_wires.ndim == 0:
            # only one control qubit
            # 1
            n_c_wires = 1
            wires.append(c_wires.item())
        elif c_wires.ndim == 1:
            # multiple control qubits
            # [1, 2]
            n_c_wires = c_wires.shape[0]
            wires.extend(list(c_wires))

        if t_wires.ndim == 0:
            # single qubit U on one set
            # 2
            n_t_wires = 1
            n_set_t_wires = 1
            wires.append(t_wires.item())
        elif t_wires.ndim == 1:
            # single qubit U on multiple sets
            # [1, 2, 3]
            # or multi qubit U on one set
            # [2, 3]
            n_t_wires = t_wires.shape[0]
            n_set_t_wires = n_t_wires // orig_u_n_wires
            wires.extend(list(t_wires.flatten()))

        elif t_wires.ndim == 2:
            # multi qubit unitary on multiple sets
            # [[2, 3], [4, 5]]
            n_t_wires = t_wires.flatten().shape[0]
            n_set_t_wires = n_t_wires // orig_u_n_wires
            wires.extend(list(t_wires.flatten()))

        n_wires = n_c_wires + n_t_wires

        # compute the new unitary, then permute
        unitary = torch.tensor(torch.zeros(2**n_wires, 2**n_wires, dtype=C_DTYPE))
        for k in range(2**n_wires - 2**n_t_wires):
            unitary[k, k] = 1.0 + 0.0j

        # compute kronecker product of all the controlled target

        controlled_u = None
        for k in range(n_set_t_wires):
            if controlled_u is None:
                controlled_u = orig_u
            else:
                controlled_u = torch.kron(controlled_u, orig_u)

        d_controlled_u = controlled_u.shape[-1]
        unitary[-d_controlled_u:, -d_controlled_u:] = controlled_u

        return cls(
            has_params=True,
            trainable=trainable,
            init_params=unitary,
            n_wires=n_wires,
            wires=wires,
        )

    @classmethod
    def _matrix(cls, params):
        return tqf.qubitunitaryfast_matrix(params)

    def build_params(self, trainable):
        return None

    def reset_params(self, init_params=None):
        self.params = torch.tensor(init_params, dtype=C_DTYPE)
        self.register_buffer(f"{self.name}_unitary", self.params)


class MultiCNOT(Operation, metaclass=ABCMeta):
    """Class for Multi qubit CNOT gate."""

    num_params = 0
    num_wires = AnyWires
    func = staticmethod(tqf.multicnot)

    @classmethod
    def _matrix(cls, params, n_wires):
        return tqf.multicnot_matrix(n_wires)

    @property
    def matrix(self):
        op_matrix = self._matrix(self.params, self.n_wires)
        return op_matrix


class MultiXCNOT(Operation, metaclass=ABCMeta):
    """Class for Multi qubit XCNOT gate."""

    num_params = 0
    num_wires = AnyWires
    func = staticmethod(tqf.multixcnot)

    @classmethod
    def _matrix(cls, params, n_wires):
        return tqf.multixcnot_matrix(n_wires)

    @property
    def matrix(self):
        op_matrix = self._matrix(self.params, self.n_wires)
        return op_matrix


class Reset(Operator, metaclass=ABCMeta):
    """Class for Reset gate."""

    num_params = 0
    num_wires = AnyWires
    func = staticmethod(tqf.reset)

    @classmethod
    def _matrix(cls, params):
        return None


class SingleExcitation(Operator, metaclass=ABCMeta):
    """Class for SingleExcitation gate."""

    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.singleexcitation)

    @classmethod
    def _matrix(cls, params):
        return tqf.singleexcitation_matrix(params)


class ECR(Operation, metaclass=ABCMeta):
    """Class for Echoed Cross Resonance Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["ecr"]
    func = staticmethod(tqf.ecr)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class QFT(Observable, metaclass=ABCMeta):
    """Class for Quantum Fourier Transform."""

    num_params = 0
    num_wires = AnyWires
    func = staticmethod(tqf.qft)

    @classmethod
    def _matrix(cls, params, n_wires):
        return tqf.qft_matrix(n_wires)


class SDG(Operation, metaclass=ABCMeta):
    """Class for SDG Gate."""

    num_params = 0
    num_wires = 1

    matrix = mat_dict["sdg"]
    func = staticmethod(tqf.sdg)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class TDG(Operation, metaclass=ABCMeta):
    """Class for TDG Gate."""

    num_params = 0
    num_wires = 1
    matrix = mat_dict["tdg"]
    func = staticmethod(tqf.tdg)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class SXDG(Operation, metaclass=ABCMeta):
    """Class for SXDG Gate."""

    num_params = 0
    num_wires = 1
    matrix = mat_dict["sxdg"]
    func = staticmethod(tqf.sxdg)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class CCZ(Operation, metaclass=ABCMeta):
    """Class for CCZ Gate."""

    num_params = 0
    num_wires = 3
    matrix = mat_dict["ccz"]
    func = staticmethod(tqf.ccz)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class ISWAP(Operation, metaclass=ABCMeta):
    """Class for ISWAP Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["iswap"]
    func = staticmethod(tqf.iswap)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class CS(Operation, metaclass=ABCMeta):
    """Class for CS Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["cs"]
    eigvals = np.array([1, 1, 1, 1j])
    func = staticmethod(tqf.cs)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class CSDG(DiagonalOperation, metaclass=ABCMeta):
    """Class for CS Dagger Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["csdg"]
    eigvals = np.array([1, 1, 1, -1j])
    func = staticmethod(tqf.csdg)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class CSX(Operation, metaclass=ABCMeta):
    """Class for CSX Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["csx"]
    func = staticmethod(tqf.csx)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class CHadamard(Operation, metaclass=ABCMeta):
    """Class for CHadamard Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["chadamard"]
    func = staticmethod(tqf.chadamard)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class CCZ(DiagonalOperation, metaclass=ABCMeta):
    """Class for CCZ Gate."""

    num_params = 0
    num_wires = 3
    matrix = mat_dict["ccz"]
    eigvals = np.array([1, 1, 1, 1, 1, 1, 1, -1])
    func = staticmethod(tqf.ccz)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class DCX(Operation, metaclass=ABCMeta):
    """Class for DCX Gate."""

    num_params = 0
    num_wires = 2
    matrix = mat_dict["dcx"]
    func = staticmethod(tqf.dcx)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class XXMINYY(Operation, metaclass=ABCMeta):
    """Class for XXMinusYY gate."""

    num_params = 2
    num_wires = 2
    func = staticmethod(tqf.xxminyy_matrix)

    @classmethod
    def _matrix(cls, params):
        return tqf.xxminyy_matrix(params)


class XXPLUSYY(Operation, metaclass=ABCMeta):
    """Class for XXPlusYY gate."""

    num_params = 2
    num_wires = 2
    func = staticmethod(tqf.xxplusyy_matrix)

    @classmethod
    def _matrix(cls, params):
        return tqf.xxplusyy_matrix(params)


class C3X(Operation, metaclass=ABCMeta):
    """Class for C3X gate."""

    num_params = 0
    num_wires = 4
    matrix = mat_dict["c3x"]
    func = staticmethod(tqf.c3x)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class R(DiagonalOperation, metaclass=ABCMeta):
    """Class for R Gate."""

    num_params = 2
    num_wires = 1
    func = staticmethod(tqf.r)

    @classmethod
    def _matrix(cls, params):
        return tqf.r_matrix(params)


class C4X(Operation, metaclass=ABCMeta):
    """Class for C4X Gate."""

    num_params = 0
    num_wires = 5
    matrix = mat_dict["c4x"]
    func = staticmethod(tqf.c4x)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class RC3X(Operation, metaclass=ABCMeta):
    """Class for RC3X Gate."""

    num_params = 0
    num_wires = 4
    matrix = mat_dict["rc3x"]
    func = staticmethod(tqf.rc3x)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class RCCX(Operation, metaclass=ABCMeta):
    """Class for RCCX Gate."""

    num_params = 0
    num_wires = 3
    matrix = mat_dict["rccx"]
    func = staticmethod(tqf.rccx)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class GlobalPhase(Operation, metaclass=ABCMeta):
    """Class for Global Phase gate."""

    num_params = 1
    num_wires = 0
    func = staticmethod(tqf.globalphase)

    @classmethod
    def _matrix(cls, params):
        return tqf.globalphase_matrix(params)


class C3SX(Operation, metaclass=ABCMeta):
    """Class for C3SX Gate."""

    num_params = 0
    num_wires = 4
    matrix = mat_dict["c3sx"]
    func = staticmethod(tqf.c3sx)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


H = Hadamard
SH = SHadamard
EchoedCrossResonance = ECR
CH = CHadamard

op_name_dict = {
    "hadamard": Hadamard,
    "h": Hadamard,
    "shadamard": SHadamard,
    "sh": SHadamard,
    "paulix": PauliX,
    "x": PauliX,
    "pauliy": PauliY,
    "y": PauliY,
    "pauliz": PauliZ,
    "z": PauliZ,
    "i": I,
    "s": S,
    "t": T,
    "sx": SX,
    "cx": CNOT,
    "cnot": CNOT,
    "cz": CZ,
    "cy": CY,
    "rx": RX,
    "ry": RY,
    "rz": RZ,
    "rxx": RXX,
    "xx": RXX,
    "ryy": RYY,
    "yy": RYY,
    "rzz": RZZ,
    "zz": RZZ,
    "rzx": RZX,
    "zx": RZX,
    "swap": SWAP,
    "sswap": SSWAP,
    "cswap": CSWAP,
    "toffoli": Toffoli,
    "ccx": Toffoli,
    "phaseshift": PhaseShift,
    "rot": Rot,
    "multirz": MultiRZ,
    "crx": CRX,
    "cry": CRY,
    "crz": CRZ,
    "crot": CRot,
    "u1": U1,
    "p": U1,
    "u2": U2,
    "u3": U3,
    "u": U3,
    "cu1": CU1,
    "cp": CU1,
    "cr": CU1,
    "cphase": CU1,
    "cu2": CU2,
    "cu3": CU3,
    "cu": CU,
    "qubitunitary": QubitUnitary,
    "qubitunitarystrict": QubitUnitaryFast,
    "qubitunitaryfast": QubitUnitaryFast,
    "trainableunitary": TrainableUnitary,
    "trainableunitarystrict": TrainableUnitaryStrict,
    "multicnot": MultiCNOT,
    "multixcnot": MultiXCNOT,
    "reset": Reset,
    "singleexcitation": SingleExcitation,
    "ecr": ECR,
    "echoedcrossresonance": ECR,
    "QFT": QFT,
    "sdg": SDG,
    "cs": CS,
    "chadamard": CHadamard,
    "ch": CH,
    "dcx": DCX,
    "xxminyy": XXMINYY,
    "xxplusyy": XXPLUSYY,
    "c3x": C3X,
    "tdg": TDG,
    "sxdg": SXDG,
    "ch": CH,
    "ccz": CCZ,
    "iswap": ISWAP,
    "csdg": CSDG,
    "csx": CSX,
    "r": R,
    "c3sx": C3SX,
    "globalphase": GlobalPhase,
    "rccx": RCCX,
    "rc3x": RC3X,
    "c4x": C4X,
}
