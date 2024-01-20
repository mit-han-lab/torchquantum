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
import torchquantum.functional as tqf
import numpy as np

from typing import Iterable
from torchquantum.plugin.qiskit import QISKIT_INCOMPATIBLE_FUNC_NAMES
from torchpack.utils.logging import logger

__all__ = [
    "RandomLayer",
    "RandomLayerAllTypes",
    "RandomOp1All",
]


class RandomOp1All(tq.QuantumModule):
    def __init__(
        self,
        n_wires: int,
        op_types=(tq.RX, tq.RY, tq.RZ),
        op_ratios=None,
        has_params=True,
        trainable=True,
        seed=None,
    ):
        """Layer adding a random gate to all wires

        Params:
            n_wires (int): number of wires/gates in integer format
            op_types (Iterable): single-wire gates to select from in iterable format
            op_ratios (Iterable): probabilities to select each gate option in iterable format
            seed (int): random seed in integer format
        """
        super().__init__()
        self.n_wires = n_wires
        self.op_types = op_types
        self.op_ratios = op_ratios
        self.seed = seed
        self.gate_all = nn.ModuleList()

        if seed is not None:
            np.random.seed(seed)

        for k in range(self.n_wires):
            op = np.random.choice(self.op_types, p=self.op_ratios)
            self.gate_all.append(op(has_params=has_params, trainable=trainable))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        for k in range(self.n_wires):
            self.gate_all[k](q_device, wires=k)


class RandomLayer(tq.QuantumModule):
    """
    Quantum module that represents a random layer of quantum operations applied to specified wires.

    Args:
        wires (int or Iterable[int]): Indices of the wires the operations are applied to.
        n_ops (int): Number of random operations in the layer.
        n_params (int): Number of parameters for each random operation.
        op_ratios (list or float): Ratios determining the relative frequencies of different operation types.
        op_types (tuple or tq.Operator): Types of random operations to be included in the layer.
        seed (int): Seed for random number generation.
        qiskit_compatible (bool): Flag indicating whether the layer should be compatible with Qiskit.

    Attributes:
        n_ops (int): Number of random operations in the layer.
        n_params (int): Number of parameters for each random operation.
        wires (list): Indices of the wires the operations are applied to.
        n_wires (int): Number of wires.
        op_types (list): Types of random operations included in the layer.
        op_ratios (numpy.array): Ratios determining the relative frequencies of different operation types.
        seed (int): Seed for random number generation.
        op_list (tq.QuantumModuleList): List of random operations in the layer.

    """

    def __init__(
        self,
        wires,
        n_ops=None,
        n_params=None,
        op_ratios=None,
        op_types=(tq.RX, tq.RY, tq.RZ, tq.CNOT),
        seed=None,
        qiskit_compatible=False,
    ):
        super().__init__()
        self.n_ops = n_ops
        self.n_params = n_params
        assert n_params is not None or n_ops is not None
        self.wires = wires if isinstance(wires, Iterable) else [wires]
        self.n_wires = len(wires)

        op_types = op_types if isinstance(op_types, Iterable) else [op_types]
        if op_ratios is None:
            op_ratios = [1] * len(op_types)
        else:
            op_ratios = op_ratios if isinstance(op_ratios, Iterable) else [op_ratios]
        op_types_valid = []
        op_ratios_valid = []

        if qiskit_compatible:
            for op_type, op_ratio in zip(op_types, op_ratios):
                if op_type().name.lower() in QISKIT_INCOMPATIBLE_FUNC_NAMES:
                    logger.warning(
                        f"Remove {op_type} from op_types to make "
                        f"the layer qiskit-compatible."
                    )
                else:
                    op_types_valid.append(op_type)
                    op_ratios_valid.append(op_ratio)
        else:
            op_types_valid = op_types
            op_ratios_valid = op_ratios

        self.op_types = op_types_valid
        self.op_ratios = np.array(op_ratios_valid) / sum(op_ratios_valid)

        self.seed = seed
        self.op_list = tq.QuantumModuleList()
        if seed is not None:
            np.random.seed(seed)
        self.build_random_layer()

    def rebuild_random_layer_from_op_list(self, n_ops_in, wires_in, op_list_in):
        """
        Rebuilds a random layer from the given operation list.
        This method is used for loading a random layer from a checkpoint.

        Args:
            n_ops_in (int): Number of operations in the layer.
            wires_in (list): Indices of the wires the operations are applied to.
            op_list_in (list): List of operations in the layer.

        """

        self.n_ops = n_ops_in
        self.wires = wires_in
        self.op_list = tq.QuantumModuleList()
        for op_in in op_list_in:
            op = tq.op_name_dict[op_in.name.lower()](
                has_params=op_in.has_params,
                trainable=op_in.trainable,
                wires=op_in.wires,
                n_wires=op_in.n_wires,
            )
            self.op_list.append(op)

    def build_random_layer(self):
        op_cnt = 0
        param_cnt = 0
        while True:
            op = np.random.choice(self.op_types, p=self.op_ratios)
            n_op_wires = op.num_wires
            if n_op_wires > self.n_wires:
                continue
            if n_op_wires == -1:
                is_AnyWire = True
                n_op_wires = self.n_wires
            else:
                is_AnyWire = False

            op_wires = list(
                np.random.choice(self.wires, size=n_op_wires, replace=False)
            )
            if is_AnyWire:
                if op().name in ["MultiRZ"]:
                    operation = op(
                        has_params=True,
                        trainable=True,
                        n_wires=n_op_wires,
                        wires=op_wires,
                    )
                else:
                    operation = op(n_wires=n_op_wires, wires=op_wires)
            elif op().name in tq.operator.parameterized_ops:
                operation = op(has_params=True, trainable=True, wires=op_wires)
            else:
                operation = op(wires=op_wires)
            self.op_list.append(operation)
            op_cnt += 1
            param_cnt += op.num_params

            if self.n_ops is not None and op_cnt == self.n_ops:
                break
            elif self.n_ops is None and self.n_params is not None:
                if param_cnt == self.n_params:
                    break
                elif param_cnt > self.n_params:
                    """
                    the last operation has too many params and exceed the
                    constraint, so need to remove it and sample another
                    """
                    op_cnt -= 1
                    param_cnt -= op.num_params
                    del self.op_list[-1]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        for op in self.op_list:
            op(q_device)


class RandomLayerAllTypes(RandomLayer):
    """
    Random layer with a wide range of quantum gate types.

    This class extends the `RandomLayer` class to include a variety of quantum gate types as options for the random layer.

    Args:
        wires (int or list): Indices of the wires the operations are applied to.
        n_ops (int): Number of operations in the layer.
        n_params (int): Number of parameters for each operation.
        op_ratios (list): Ratios for selecting different types of operations.
        op_types (tuple): Types of operations to include in the layer.
        seed (int): Seed for the random number generator.
        qiskit_compatible (bool): Flag indicating whether the layer should be Qiskit-compatible.

    """

    def __init__(
        self,
        wires,
        n_ops=None,
        n_params=None,
        op_ratios=None,
        op_types=(
            tq.Hadamard,
            tq.SHadamard,
            tq.PauliX,
            tq.PauliY,
            tq.PauliZ,
            tq.S,
            tq.T,
            tq.SX,
            tq.CNOT,
            tq.CZ,
            tq.CY,
            tq.RX,
            tq.RY,
            tq.RZ,
            tq.RZZ,
            tq.SWAP,
            tq.CSWAP,
            tq.Toffoli,
            tq.PhaseShift,
            tq.Rot,
            tq.MultiRZ,
            tq.CRX,
            tq.CRY,
            tq.CRZ,
            tq.CRot,
            tq.U1,
            tq.U2,
            tq.U3,
            tq.MultiCNOT,
            tq.MultiXCNOT,
        ),
        seed=None,
        qiskit_compatible=False,
    ):
        super().__init__(
            wires=wires,
            n_ops=n_ops,
            n_params=n_params,
            op_ratios=op_ratios,
            op_types=op_types,
            seed=seed,
            qiskit_compatible=qiskit_compatible,
        )
