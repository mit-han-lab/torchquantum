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

from .layers import LayerTemplate0, Op1QAllLayer
from ..entanglement.op2_layer import Op2QAllLayer

class SethLayer0(LayerTemplate0):
    """
    Layer template with Seth blocks.

    This layer template consists of Seth blocks, which include RZZ and RY gates, repeated for the specified number of blocks.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the Seth layers for the template.
        forward: Applies the quantum layer to the given quantum device.

    """

    name = "seth_0"

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
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
        return layers_all


class SethLayer1(LayerTemplate0):
    """
    Layer template with extended Seth blocks.

    This layer template consists of extended Seth blocks, which include RZZ and RY gates repeated twice for each block.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the extended Seth layers for the template.
        forward: Applies the quantum layer to the given quantum device.

    """

    name = "seth_1"

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
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
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


class SethLayer2(LayerTemplate0):
    """
    Layer template with Seth blocks using Op2QFit32Layer.

    This layer template consists of Seth blocks using the Op2QFit32Layer, which includes RZZ gates and supports 32 wires.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the Seth layers with Op2QFit32Layer for the template.
        forward: Applies the quantum layer to the given quantum device.

    """

    name = "seth_2"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch["n_blocks"]):
            layers_all.append(
                Op2QFit32Layer(
                    op=tq.RZZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True,
                )
            )
        return layers_all
