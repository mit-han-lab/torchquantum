import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

from enum import IntEnum
from torchquantum.functional import mat_dict
from torchquantum.quantization.clifford_quantization import CliffordQuantizer
from abc import ABCMeta
from .macro import C_DTYPE, F_DTYPE
from torchpack.utils.logging import logger
from typing import Iterable, Union, List

__all__ = [
    'op_name_dict',
    'Operator',
    'Operation',
    'DiagonalOperation',
    'Observable',
    'Hadamard',
    'H',
    'SHadamard',
    'PauliX',
    'PauliY',
    'PauliZ',
    'I',
    'S',
    'T',
    'SX',
    'CNOT',
    'CZ',
    'CY',
    'RX',
    'RY',
    'RZ',
    'RXX',
    'RYY',
    'RZZ',
    'RZX',
    'SWAP',
    'SSWAP',
    'CSWAP',
    'Toffoli',
    'PhaseShift',
    'Rot',
    'MultiRZ',
    'CRX',
    'CRY',
    'CRZ',
    'CRot',
    'U1',
    'U2',
    'U3',
    'CU1',
    'CU2',
    'CU3',
    'QubitUnitary',
    'QubitUnitaryFast',
    'TrainableUnitary',
    'TrainableUnitaryStrict',
    'MultiCNOT',
    'MultiXCNOT',
    'Reset',
    'SingleExcitation',
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


class Operator(tq.QuantumModule):
    """The class for quantum operators."""
    fixed_ops = [
        'Hadamard',
        'SHadamard',
        'PauliX',
        'PauliY',
        'PauliZ',
        'I',
        'S',
        'T',
        'SX',
        'CNOT',
        'CZ',
        'CY',
        'SWAP',
        'SSWAP',
        'CSWAP',
        'Toffoli',
        'MultiCNOT',
        'MultiXCNOT',
        'Reset',
    ]

    parameterized_ops = [
        'RX',
        'RY',
        'RZ',
        'RXX',
        'RYY',
        'RZZ',
        'RZX',
        'PhaseShift',
        'Rot',
        'MultiRZ',
        'CRX',
        'CRY',
        'CRZ',
        'CRot',
        'U1',
        'U2',
        'U3',
        'CU1',
        'CU2',
        'CU3',
        'QubitUnitary',
        'QubitUnitaryFast',
        'TrainableUnitary',
        'TrainableUnitaryStrict',
        'SingleExcitation',
    ]

    @property
    def name(self):
        """String for the name of the operator."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name of the operator.

        Args:
            value (str): operator name.

        """
        self._name = value

    def __init__(self,
                 has_params: bool = False,
                 trainable: bool = False,
                 init_params=None,
                 n_wires=None,
                 wires=None):
        """__init__ function for Operator.

        Args:
            has_params (bool, optional): Whether the operations has parameters.
                Defaults to False.
            trainable (bool, optional): Whether the parameters are trainable
                (if contains parameters). Defaults to False.
            init_params (torch.Tensor, optional): Initial parameters.
                Defaults to None.
            n_wires (int, optional): Number of qubits. Defaults to None.
            wires (Union[int, List[int]], optional): Which qubit the operation
                is applied to. Defaults to None.
        """
        super().__init__()
        self.params = None
        # number of wires of the operator
        # n_wires is used in gates that can be applied to arbitrary number
        # of qubits such as MultiRZ
        self.n_wires = n_wires
        # wires that the operator applies to
        self.wires = wires
        self._name = self.__class__.__name__
        # for static mode
        self.static_matrix = None
        self.inverse = False
        self.clifford_quantization = False

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

    @classmethod
    def _matrix(cls, params):
        """The unitary matrix of the operator.

        Args:
            params (torch.Tensor, optional): The parameters for parameterized
                operators.

        Returns: None.

        """
        raise NotImplementedError

    @property
    def matrix(self):
        """The unitary matrix of the operator."""
        return self._matrix(self.params)

    @classmethod
    def _eigvals(cls, params):
        """The eigenvalues of the unitary matrix of the operator.

        Args:
            params (torch.Tensor, optional): The parameters for parameterized
                operators.

        Returns: None.

        """
        raise NotImplementedError

    @property
    def eigvals(self):
        """The eigenvalues of the unitary matrix of the operator.

        Returns: Eigenvalues.

        """
        return self._eigvals(self.params)

    def _get_unitary_matrix(self):
        """Obtain the unitary matrix of the operator.

        Returns: Unitary matrix.

        """
        return self.matrix

    def set_wires(self, wires):
        """Set which qubits the operator is applied to.

        Args:
            wires (Union[int, List[int]]): Qubits the operator is applied to.

        Returns: None.

        """
        self.wires = [wires] if isinstance(wires, int) else wires

    def forward(self, q_device: tq.QuantumDevice, wires=None, params=None,
                inverse=False):
        """Apply the operator to the quantum device states.

        Args:
            q_device (torchquantum.QuantumDevice): Quantum Device that the
                operator is applied to.
            wires (Union[int, List[int]]): Qubits that the operator is
                applied to.
            params (torch.Tensor): Parameters of the operator
            inverse (bool): Whether inverse the unitary matrix of the operator.

        Returns:

        """
        # try:
        #     assert self.name in self.fixed_ops or \
        #            self.has_params ^ (params is not None)
        # except AssertionError as err:
        #     logger.exception(f"Parameterized gate either has its "
        #                      f"own parameters or has input as parameters")
        #     raise err

        # try:
        #     assert not (self.wires is None and wires is None)
        # except AssertionError as err:
        #     logger.exception(f"Need to specify the wires either when "
        #                      f"initialize or when forward")
        #     raise err

        if params is not None:
            self.params = params

        if self.params is not None:
            self.params = self.params.unsqueeze(-1) if self.params.dim() == 1 \
                else self.params

        if wires is not None:
            # update the wires
            wires = [wires] if isinstance(wires, int) else wires
            self.wires = wires

        self.inverse = inverse

        if self.static_mode:
            self.parent_graph.add_op(self)
            return

        # non-parameterized gate
        if self.params is None:
            if self.n_wires is None:
                self.func(q_device, self.wires, inverse=inverse)
            else:
                self.func(q_device, self.wires, n_wires=self.n_wires,
                          inverse=inverse)
        else:
            if isinstance(self.noise_model_tq, tq.NoiseModelTQPhase):
                params = self.noise_model_tq.add_noise(self.params)
            else:
                params = self.params

            if self.clifford_quantization:
                params = CliffordQuantizer.quantize_sse(params)
            if self.n_wires is None:
                self.func(q_device, self.wires, params=params,
                          inverse=inverse)
            else:
                self.func(q_device, self.wires, params=params,
                          n_wires=self.n_wires, inverse=inverse)

        if self.noise_model_tq is not None and \
                self.noise_model_tq.is_add_noise:
            noise_ops = self.noise_model_tq.sample_noise_op(self)
            if len(noise_ops):
                for noise_op in noise_ops:
                    noise_op(q_device)


class Observable(Operator, metaclass=ABCMeta):
    """Class for Observables.

    """
    def __init__(self,
                 has_params: bool = False,
                 trainable: bool = False,
                 init_params=None,
                 n_wires=None,
                 wires=None):
        """Init function of the Observable class

        has_params (bool, optional): Whether the operations has parameters.
                Defaults to False.
            trainable (bool, optional): Whether the parameters are trainable
                (if contains parameters). Defaults to False.
            init_params (torch.Tensor, optional): Initial parameters.
                Defaults to None.
            n_wires (int, optional): Number of qubits. Defaults to None.
            wires (Union[int, List[int]], optional): Which qubit the operation
                is applied to. Defaults to None.
        """
        super().__init__(
            has_params=has_params,
            trainable=trainable,
            init_params=init_params,
            n_wires=n_wires,
            wires=wires
        )
        self.return_type = None

    def diagonalizing_gates(self):
        """The diagonalizing gates when perform measurements.

        Returns: None.

        """
        raise NotImplementedError


class Operation(Operator, metaclass=ABCMeta):
    """_summary_"""
    def __init__(self,
                 has_params: bool = False,
                 trainable: bool = False,
                 init_params=None,
                 n_wires=None,
                 wires=None):
        """_summary_

        Args:
            has_params (bool, optional): Whether the operations has parameters.
                Defaults to False.
            trainable (bool, optional): Whether the parameters are trainable
                (if contains parameters). Defaults to False.
            init_params (torch.Tensor, optional): Initial parameters.
                Defaults to None.
            n_wires (int, optional): Number of qubits. Defaults to None.
            wires (Union[int, List[int]], optional): Which qubit the operation is applied to.
                Defaults to None.
        """
        super().__init__(
            has_params=has_params,
            trainable=trainable,
            init_params=init_params,
            n_wires=n_wires,
            wires=wires
        )
        if type(self.num_wires) == int:
            self.n_wires = self.num_wires

    @property
    def matrix(self):
        """The unitary matrix of the operator."""
        op_matrix = self._matrix(self.params)

        return op_matrix

    @property
    def eigvals(self):
        """"The eigenvalues of the unitary matrix of the operator.

        Returns:
            torch.Tensor: Eigenvalues.

        """
        op_eigvals = self._eigvals(self.params)

        return op_eigvals

    def init_params(self):
        """Initialize the parameters.

        Raises:
            NotImplementedError: The init param function is not implemented.
        """
        raise NotImplementedError

    def build_params(self, trainable):
        """Build parameters.

        Args:
            trainable (bool): Whether the parameters are trainable.

        Returns:
            torch.Tensor: Built parameters.
        """
        parameters = nn.Parameter(torch.empty([1, self.num_params],
                                              dtype=F_DTYPE))
        parameters.requires_grad = True if trainable else False
        self.register_parameter(f"{self.name}_params", parameters)
        return parameters

    def reset_params(self, init_params=None):
        """Reset parameters.

        Args:
            init_params (torch.Tensor, optional): Input the initialization
                parameters. Defaults to None.
        """
        if init_params is not None:
            if isinstance(init_params, Iterable):
                for k, init_param in enumerate(init_params):
                    torch.nn.init.constant_(self.params[:, k], init_param)
            else:
                torch.nn.init.constant_(self.params, init_params)
        else:
            torch.nn.init.uniform_(self.params, -np.pi, np.pi)


class DiagonalOperation(Operation, metaclass=ABCMeta):
    """Class for Diagonal Operation."""
    @classmethod
    def _eigvals(cls, params):
        """The eigenvalues of the unitary matrix of the operator.

        Args:
            params (torch.Tensor, optional): The parameters for parameterized
                operators.

        Returns: None.
        raise NotImplementedError
    """

    @property
    def eigvals(self):
        """The eigenvalues of the unitary matrix of the operator.

        Returns: Eigenvalues.

        """
        return super().eigvals

    @classmethod
    def _matrix(cls, params):
        """The unitary matrix of the operator.

        Args:
            params (torch.Tensor, optional): The parameters for parameterized
                operators.

        Returns: None.

        """
        return torch.diag(cls._eigvals(params))


class Hadamard(Observable, metaclass=ABCMeta):
    """Class for Hadamard Gate."""
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = mat_dict['hadamard']
    func = staticmethod(tqf.hadamard)

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


class SHadamard(Operation, metaclass=ABCMeta):
    """Class for SHadamard Gate."""
    num_params = 0
    num_wires = 1
    matrix = mat_dict['shadamard']
    func = staticmethod(tqf.shadamard)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class PauliX(Observable, metaclass=ABCMeta):
    """Class for Pauli X Gate."""
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = mat_dict['paulix']
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
    """Class for Pauli Y Gate."""
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = mat_dict['pauliy']
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
    """Class for Pauli Z Gate."""
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = mat_dict['pauliz']
    func = staticmethod(tqf.pauliz)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return []


class I(Observable, metaclass=ABCMeta):
    """Class for Identity Gate."""
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1], dtype=C_DTYPE)
    matrix = mat_dict['i']
    func = staticmethod(tqf.i)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return []


class S(DiagonalOperation, metaclass=ABCMeta):
    """Class for S Gate."""
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1j], dtype=C_DTYPE)
    matrix = mat_dict['s']
    func = staticmethod(tqf.s)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class T(DiagonalOperation, metaclass=ABCMeta):
    """Class for T Gate."""
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1j], dtype=C_DTYPE)
    matrix = mat_dict['t']
    func = staticmethod(tqf.t)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class SX(Operation, metaclass=ABCMeta):
    """Class for SX Gate."""
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1j], dtype=C_DTYPE)
    matrix = mat_dict['sx']
    func = staticmethod(tqf.sx)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class CNOT(Operation, metaclass=ABCMeta):
    """Class for CNOT Gate."""
    num_params = 0
    num_wires = 2
    matrix = mat_dict['cnot']
    func = staticmethod(tqf.cnot)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class CZ(DiagonalOperation, metaclass=ABCMeta):
    """Class for CZ Gate."""
    num_params = 0
    num_wires = 2
    eigvals = np.array([1, 1, 1, -1])
    matrix = mat_dict['cz']
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
    matrix = mat_dict['cy']
    func = staticmethod(tqf.cy)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class SWAP(Operation, metaclass=ABCMeta):
    """Class for SWAP Gate."""
    num_params = 0
    num_wires = 2
    matrix = mat_dict['swap']
    func = staticmethod(tqf.swap)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class SSWAP(Operation, metaclass=ABCMeta):
    """Class for SSWAP Gate."""
    num_params = 0
    num_wires = 2
    matrix = mat_dict['sswap']
    func = staticmethod(tqf.sswap)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class CSWAP(Operation, metaclass=ABCMeta):
    """Class for CSWAP Gate."""
    num_params = 0
    num_wires = 3
    matrix = mat_dict['cswap']
    func = staticmethod(tqf.cswap)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class Toffoli(Operation, metaclass=ABCMeta):
    """Class for Toffoli Gate."""
    num_params = 0
    num_wires = 3
    matrix = mat_dict['toffoli']
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
        parameters = nn.Parameter(torch.empty(
            1, 2 ** self.n_wires, 2 ** self.n_wires, dtype=C_DTYPE))
        parameters.requires_grad = True if trainable else False
        self.register_parameter(f"{self.name}_params", parameters)
        return parameters

    def reset_params(self, init_params=None):
        """Reset the parameters.

        Args:
            init_params (torch.Tensor, optional): Initial parameters.

        Returns:
            None.

        """
        mat = torch.randn((1, 2 ** self.n_wires, 2 ** self.n_wires),
                          dtype=C_DTYPE)
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

    def __init__(self,
                 has_params: bool = False,
                 trainable: bool = False,
                 init_params=None,
                 n_wires=None,
                 wires=None):
        super().__init__(
            has_params=True,
            trainable=trainable,
            init_params=init_params,
            n_wires=n_wires,
            wires=wires
        )

    @classmethod
    def from_controlled_operation(cls,
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
            unitary[k, k] = 1. + 0.j

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
    func = staticmethod(tqf.single_excitation)

    @classmethod
    def _matrix(cls, params):
        return tqf.single_excitation_matrix(params)


H=Hadamard


op_name_dict = {
    'hadamard': Hadamard,
    'h': Hadamard,
    'shadamard': SHadamard,
    'sh': SHadamard,
    'paulix': PauliX,
    'x': PauliX,
    'pauliy': PauliY,
    'y': PauliY,
    'pauliz': PauliZ,
    'z': PauliZ,
    'i': I,
    's': S,
    't': T,
    'sx': SX,
    'cx': CNOT,
    'cnot': CNOT,
    'cz': CZ,
    'cy': CY,
    'rx': RX,
    'ry': RY,
    'rz': RZ,
    'rxx': RXX,
    'xx': RXX,
    'ryy': RYY,
    'yy': RYY,
    'rzz': RZZ,
    'zz': RZZ,
    'rzx': RZX,
    'zx': RZX,
    'swap': SWAP,
    'sswap': SSWAP,
    'cswap': CSWAP,
    'toffoli': Toffoli,
    'ccx': Toffoli,
    'phaseshift': PhaseShift,
    'rot': Rot,
    'multirz': MultiRZ,
    'crx': CRX,
    'cry': CRY,
    'crz': CRZ,
    'crot': CRot,
    'u1': U1,
    'p': U1,
    'u2': U2,
    'u3': U3,
    'u': U3,
    'cu1': CU1,
    'cp': CU1,
    'cr': CU1,
    'cphase': CU1,
    'cu2': CU2,
    'cu3': CU3,
    'cu': CU3,
    'qubitunitary': QubitUnitary,
    'qubitunitarystrict': QubitUnitaryFast,
    'qubitunitaryfast': QubitUnitaryFast,
    'trainableunitary': TrainableUnitary,
    'trainableunitarystrict': TrainableUnitaryStrict,
    'multicnot': MultiCNOT,
    'multixcnot': MultiXCNOT,
    'reset': Reset,
}
