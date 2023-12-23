import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

from typing import Iterable
from torchquantum.plugin.qiskit import QISKIT_INCOMPATIBLE_FUNC_NAMES
from torchpack.utils.logging import logger

from .layers import LayerTemplate0


class RYRYCXLayer0(LayerTemplate0):
    """
    Layer template with RYRYCX blocks.

    This layer template consists of RYRYCX blocks using the Op1QAllLayer and CXLayer modules.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the RYRYCX layers with Op1QAllLayer and CXLayer for the template.
        forward: Applies the quantum layer to the given quantum device.

    """

    name = "ryrycx"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch["n_blocks"]):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(CXLayer(n_wires=self.n_wires))
        return layers_all


class RYRYRYCXCXCXLayer0(LayerTemplate0):
    """
    Layer template with RYRYRYCXCXCX blocks.

    This layer template consists of RYRYRYCXCXCX blocks using the RYRYCXCXLayer module.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the RYRYRYCXCXCX layers with RYRYCXCXLayer for the template.

    """

    name = "ryryrycxcxcx"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch["n_blocks"]):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(CXCXCXLayer(n_wires=self.n_wires))
        return layers_all


class RYRYRYLayer0(LayerTemplate0):
    """
    Layer template with RYRYRYCXCXCX blocks.

    This layer template consists of RYRYRYCXCXCX blocks using the Op1QAllLayer and CXCXCXLayer modules.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the RYRYRYCXCXCX layers with Op1QAllLayer and CXCXCXLayer for the template.


    """

    name = "ryryry"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch["n_blocks"]):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
        return layers_all


class RYRYRYSWAPSWAPLayer0(LayerTemplate0):
    """
    Layer template with RYRYRYSWAPSWAP blocks.

    This layer template consists of RYRYRYSWAPSWAP blocks using the Op1QAllLayer and SWAPSWAPLayer modules.

    Args:
        arch (dict, optional): The architecture configuration for the layer. Defaults to None.

    Attributes:
        n_wires (int): The number of wires in the layer.
        arch (dict): The architecture configuration for the layer.
        n_blocks (int): The number of blocks in the layer.
        layers_all (tq.QuantumModuleList): The list of layers in the template.

    Methods:
        build_layers: Builds the RYRYRYSWAPSWAP layers with Op1QAllLayer and SWAPSWAPLayer for the template.


    """

    name = "ryryryswapswap"

    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch["n_blocks"]):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY, n_wires=self.n_wires, has_params=True, trainable=True
                )
            )
            layers_all.append(SWAPSWAPLayer(n_wires=self.n_wires))
        return layers_all
