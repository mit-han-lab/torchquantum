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
from torchquantum.layer.entanglement.op2_layer import Op2QAllLayer

__all__ = [
    "SimpleQLayer",
    "LayerTemplate0",
    "Op1QAllLayer",
]


class SimpleQLayer(tq.QuantumModule):
    """
    Simple quantum layer consisting of three parameterized gates applied to specific wires.

    This class represents a simple quantum layer with three parameterized gates: RX, RY, and RZ. The gates are applied to specific wires in the quantum device.

    Args:
        n_wires (int): Number of wires in the quantum device.

    """

    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires
        self.gate1 = tq.RX(has_params=True, trainable=True)
        self.gate2 = tq.RY(has_params=True, trainable=True)
        self.gate3 = tq.RZ(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, q_dev):
        self.q_device = q_dev
        tqf.x(q_dev, wires=0, static=self.static_mode, parent_graph=self.graph)
        self.gate1(q_dev, wires=1)
        self.gate2(q_dev, wires=1)
        self.gate3(q_dev, wires=1)
        tqf.x(q_dev, wires=2, static=self.static_mode, parent_graph=self.graph)


class Op1QAllLayer(tq.QuantumModule):
    """
    Quantum layer applying the same single-qubit operation to all wires.

    This class represents a quantum layer that applies the same single-qubit operation to all wires in the quantum device.

    Args:
        op (tq.Operator): Single-qubit operation to be applied.
        n_wires (int): Number of wires in the quantum device.
        has_params (bool, optional): Flag indicating if the operation has parameters. Defaults to False.
        trainable (bool, optional): Flag indicating if the operation is trainable. Defaults to False.

    """

    def __init__(self, op, n_wires: int, has_params=False, trainable=False):
        super().__init__()
        self.n_wires = n_wires
        self.op = op
        self.ops_all = tq.QuantumModuleList()
        for k in range(n_wires):
            self.ops_all.append(op(has_params=has_params, trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_wires):
            self.ops_all[k](q_device, wires=k)


class LayerTemplate0(tq.QuantumModule):
    """
    A template for a custom quantum layer.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer. (Optional)
        n_layers_per_block (int): The number of layers per block. (Optional)
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Abstract method to build the layers of the template.
        forward: Applies the quantum layer to the given quantum device.

    """

    name = None

    def __init__(self, arch: dict = None):
        super().__init__()
        self.n_wires = arch["n_wires"]
        self.arch = arch

        self.n_blocks = arch.get("n_blocks", None)
        self.n_layers_per_block = arch.get("n_layers_per_block", None)

        self.layers_all = self.build_layers()

    def build_layers(self):
        raise NotImplementedError

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for k in range(len(self.layers_all)):
            self.layers_all[k](q_device)


class CXRZSXLayer0(LayerTemplate0):
    """
    Layer template with CXRZSX blocks.

    This layer template consists of CXRZSX blocks, which include RZ, CNOT, and SX gates, repeated for the specified number of blocks.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the CXRZSX layers for the template.
        forward: Applies the quantum layer to the given quantum device.

    """

    name = "cxrzsx_0"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()

        layers_all.append(
            Op1QAllLayer(
                op=tq.RZ, n_wires=self.n_wires, has_params=True, trainable=True
            )
        )
        layers_all.append(
            Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, jump=1, circular=False)
        )
        for k in range(self.arch["n_blocks"]):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RZ, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(
                Op1QAllLayer(
                    op=tq.SX, n_wires=self.n_wires, has_params=False, trainable=False
                )
            )
        layers_all.append(
            Op1QAllLayer(
                op=tq.RZ, n_wires=self.n_wires, has_params=True, trainable=True
            )
        )
        return layers_all


class RZZLayer0(LayerTemplate0):
    """
    Layer template with RZZ blocks.

    This layer template consists of RZZ blocks using the Op2QAllLayer.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the RZZ layers with Op2QAllLayer for the template.
        forward: Applies the quantum layer to the given quantum device.

    """

    name = "rzz_0"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch["n_blocks"]):
            layers_all.append(
                Op2QAllLayer(
                    op=tq.RZZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True,
                )
            )
        return layers_all


class BarrenLayer0(LayerTemplate0):
    """
    Layer template with Barren blocks.

    This layer template consists of Barren blocks using the Op1QAllLayer and Op2QAllLayer.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the Barren layers with Op1QAllLayer and Op2QAllLayer for the template.
        forward: Applies the quantum layer to the given quantum device.

    """

    name = "barren_0"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        layers_all.append(
            Op1QAllLayer(
                op=tq.SHadamard,
                n_wires=self.n_wires,
            )
        )
        for k in range(self.arch["n_blocks"]):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RX, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RZ, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires, jump=1))
        return layers_all


class FarhiLayer0(LayerTemplate0):
    """
    Layer template with Farhi blocks.

    This layer template consists of Farhi blocks using the Op2QAllLayer.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the Farhi layers with Op2QAllLayer for the template.
        forward: Applies the quantum layer to the given quantum device.

    """

    name = "farhi_0"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch["n_blocks"]):
            layers_all.append(
                Op2QAllLayer(
                    op=tq.RZX,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True,
                )
            )
            layers_all.append(
                Op2QAllLayer(
                    op=tq.RXX,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True,
                )
            )
        return layers_all


class MaxwellLayer0(LayerTemplate0):
    """
    Layer template with Maxwell blocks.

    This layer template consists of Maxwell blocks using the Op1QAllLayer and Op2QAllLayer modules.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the Maxwell layers with Op1QAllLayer and Op2QAllLayer for the template.
        forward: Applies the quantum layer to the given quantum device.

    """

    name = "maxwell_0"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch["n_blocks"]):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RX, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(Op1QAllLayer(op=tq.S, n_wires=self.n_wires))
            layers_all.append(
                Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, jump=1, circular=True)
            )

            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(Op1QAllLayer(op=tq.T, n_wires=self.n_wires))
            layers_all.append(
                Op2QAllLayer(op=tq.SWAP, n_wires=self.n_wires, jump=1, circular=True)
            )

            layers_all.append(
                Op1QAllLayer(
                    op=tq.RZ, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(Op1QAllLayer(op=tq.Hadamard, n_wires=self.n_wires))
            layers_all.append(
                Op2QAllLayer(op=tq.SSWAP, n_wires=self.n_wires, jump=1, circular=True)
            )

            layers_all.append(
                Op1QAllLayer(
                    op=tq.U1, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(
                Op2QAllLayer(
                    op=tq.CU3,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True,
                )
            )

        return layers_all


class RXYZCXLayer0(LayerTemplate0):
    """
    Layer template with RXYZCX blocks.

    This layer template consists of RXYZCX blocks using the RXYZCXLayer module.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the RXYZCX layers with RXYZCXLayer for the template.

    """

    name = "rxyzcx_0"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch["n_blocks"]):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RX, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RZ, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(
                Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, jump=1, circular=True)
            )
        return layers_all
