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

from .layers import LayerTemplate0


class SWAPSWAPLayer(tq.QuantumModule):
    """
    Quantum layer with a sequence of SWAP gates applied to two specified pairs of wires.

    This class represents a quantum layer with a sequence of SWAP gates applied to two specified pairs of wires in the quantum device.

    Args:
        n_wires (int): Number of wires in the quantum device.

    """

    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires

    @tq.static_support
    def forward(self, q_dev):
        self.q_device = q_dev
        tqf.swap(q_dev, wires=[0, 1], static=self.static_mode, parent_graph=self.graph)
        tqf.swap(q_dev, wires=[1, 2], static=self.static_mode, parent_graph=self.graph)


class SWAPSWAPLayer0(LayerTemplate0):
    """
    Layer template with SWAPSWAP blocks.

    This layer template consists of SWAPSWAP blocks using the SWAPSWAPLayer module.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the SWAPSWAP layers with SWAPSWAPLayer for the template.

    """

    name = "swapswap"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch["n_blocks"]):
            layers_all.append(SWAPSWAPLayer(n_wires=self.n_wires))
        return layers_all
