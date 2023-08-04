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
import torchquantum as tq
from torchquantum.layer.layers import (
    LayerTemplate0,
    Op1QAllLayer,
    Op2QAllLayer,
    RandomOp1All,
)

__all__ = [
    "NLocal",
    "TwoLocal",
    "ExcitationPreserving",
    "EfficientSU2",
    "RealAmplitudes",
    "PauliTwoDesign",
]


class NLocal(LayerTemplate0):
    """Layer Template for a NLocal Class

    Args:
        rotation_ops (list): gates for the rotation layer as a list of torchquantum operations
        entanglement_ops (list): gates for the entanglement layer as a list of torchquantum operations
        arch (dict): circuit architecture in a dictionary format
        rotation_layer (torchquantum.QuantumModule): type of rotation layer in a torchquantum.QuantumModule format
        entanglement_layer (torchquantum.QuantumModule): type of entanglement layer in a torchquantum.QuantumModule format
        reps (int): number of reptitions of the rotation and entanglement layers in a integer format
        rotation_layer_params (dict): additional parameters for the rotation layer in a dictionary format
        entanglement_layer_params (dict): additional parameters for the entanglement layer in a dictionary format
        initial_circuit (torchquantum.QuantumModule): initial gates or layer in a QuantumModule format
        skip_final_rotation_layer (bool): whether or not to add the final rotation layer as a boolean
    """

    def __init__(
        self,
        rotation_ops: list = None,
        entanglement_ops: list = None,
        arch: dict = None,
        rotation_layer: tq.QuantumModule = Op1QAllLayer,
        entanglement_layer: tq.QuantumModule = Op2QAllLayer,
        reps: int = 1,
        rotation_layer_params: dict = {},
        entanglement_layer_params: dict = {},
        initial_circuit: tq.QuantumModule = None,
        skip_final_rotation_layer: bool = False,
    ):
        # rotation block options
        self.rotation_ops = rotation_ops
        self.rotation_layer = rotation_layer
        self.rotation_layer_params = rotation_layer_params

        # entanglement block options
        self.entanglement_ops = entanglement_ops
        self.entanglement_layer = entanglement_layer
        self.entanglement_layer_params = entanglement_layer_params

        # extra parameters
        self.initial_circuit = initial_circuit
        self.skip_final_rotation_layer = skip_final_rotation_layer
        self.reps = reps

        # initialize the LayerTemplate0
        super().__init__(arch)

    def build_initial_layer(self):
        """Build the initial layer"""
        return self.initial_circuit

    def build_rotation_block(self):
        """Build rotation block"""
        rotation_layers = []
        for rot in self.rotation_ops:
            rotation_layers.append(
                self.rotation_layer(
                    op=rot, n_wires=self.n_wires, **self.rotation_layer_params
                )
            )
        return rotation_layers

    def build_entanglement_block(self):
        """Build entanglement block"""
        entanglement_layers = []
        for entanglement in self.entanglement_ops:
            entanglement_layers.append(
                self.entanglement_layer(
                    op=entanglement,
                    n_wires=self.n_wires,
                    **self.entanglement_layer_params,
                )
            )
        return entanglement_layers

    def build_layers(self):
        """Build nlocal circuit"""
        layers_all = tq.QuantumModuleList()

        # add the initial circuit
        initial_circuit = self.build_initial_layer()
        if initial_circuit is not None:
            layers_all.append(initial_circuit)

        # repeat for each rep
        for _ in range(self.reps):
            # add rotation blocks to the qubits
            layers_all.extend(self.build_rotation_block())

            # add entanglement blocks to the qubits
            layers_all.extend(self.build_entanglement_block())

        # add final rotation layer
        if not self.skip_final_rotation_layer:
            layers_all.extend(self.build_rotation_block())

        # return QuantumModuleList
        return layers_all


class TwoLocal(NLocal):
    """Layer Template for a TwoLocal Class

    Args:
        rotation_ops (list): gates for the rotation layer as a list of torchquantum operations
        entanglement_ops (list): gates for the entanglement layer as a list of torchquantum operations
        arch (dict): circuit architecture in a dictionary format
        rotation_layer (torchquantum.QuantumModule): type of rotation layer in a torchquantum.QuantumModule format
        entanglement_layer (str): type of entanglement layer in a string ("linear", "reverse_linear", "circular", "full") or tq.QuantumModule format
        reps (int): number of reptitions of the rotation and entanglement layers in a integer format
        entanglement_layer_params (dict): additional parameters for the entanglement layer in a dictionary forma
        initial_circuit (torchquantum.QuantumModule): initial gates or layer in a QuantumModule formatt
        skip_final_rotation_layer (bool): whether or not to add the final rotation layer as a boolean
    """

    def __init__(
        self,
        rotation_ops: list = None,
        entanglement_ops: list = None,
        arch: dict = None,
        rotation_layer: tq.QuantumModule = Op1QAllLayer,
        entanglement_layer: str = "linear",
        reps: int = 1,
        entanglement_layer_params: dict = {},
        initial_circuit: tq.QuantumModule = None,
        skip_final_rotation_layer: bool = False,
    ):
        # if passed as string, determine entanglement type
        if entanglement_layer == "linear":
            entanglement_layer = Op2QAllLayer
        elif entanglement_layer == "reverse_linear":
            entanglement_layer = Op2QAllLayer
            entanglement_layer_params = {"wire_reverse": True}
        elif entanglement_layer == "circular":
            entanglement_layer = Op2QAllLayer
            entanglement_layer_params = {"circular": True}
        elif entanglement_layer == "full":
            entanglement_layer = Op2QDenseLayer

        # initialize
        super().__init__(
            arch=arch,
            rotation_ops=rotation_ops,
            rotation_layer=rotation_layer,
            rotation_layer_params={"has_params": True, "trainable": True},
            entanglement_ops=entanglement_ops,
            entanglement_layer=entanglement_layer,
            entanglement_layer_params=entanglement_layer_params,
            initial_circuit=initial_circuit,
            reps=reps,
            skip_final_rotation_layer=skip_final_rotation_layer,
        )


class ExcitationPreserving(TwoLocal):
    """Layer Template for a ExcitationPreserving circuit

    Args:
        arch (dict): circuit architecture in a dictionary format
        entanglement_layer (str): type of entanglement layer in a string ("linear", "reverse_linear", "circular", "full") or tq.QuantumModule format
        reps (int): number of reptitions of the rotation and entanglement layers in a integer format
        skip_final_rotation_layer (bool): whether or not to add the final rotation layer as a boolean
    """

    def __init__(
        self,
        arch: dict = None,
        entanglement_layer: str = "full",
        reps: int = 3,
        skip_final_rotation_layer: bool = False,
    ):
        # construct circuit with rotation layers of RZ and entanglement with RXX and RYY
        super().__init__(
            arch=arch,
            rotation_ops=[tq.RZ],
            entanglement_ops=[tq.RXX, tq.RYY],
            entanglement_layer=entanglement_layer,
            entanglement_layer_params={"has_params": True, "trainable": True},
            reps=reps,
            skip_final_rotation_layer=skip_final_rotation_layer,
        )


class EfficientSU2(TwoLocal):
    """Layer Template for a EfficientSU2 circuit

    Args:
        arch (dict): circuit architecture in a dictionary format
        entanglement_layer (str): type of entanglement layer in a string ("linear", "reverse_linear", "circular", "full") or tq.QuantumModule format
        reps (int): number of reptitions of the rotation and entanglement layers in a integer format
        skip_final_rotation_layer (bool): whether or not to add the final rotation layer as a boolean
    """

    def __init__(
        self,
        arch: dict = None,
        entanglement_layer: str = "reverse_linear",
        reps: int = 3,
        skip_final_rotation_layer: bool = False,
    ):
        # construct circuit with rotation layers of RY and RZ and entanglement with CX
        super().__init__(
            arch=arch,
            rotation_ops=[tq.RY, tq.RZ],
            entanglement_ops=[tq.CNOT],
            entanglement_layer=entanglement_layer,
            reps=reps,
            skip_final_rotation_layer=skip_final_rotation_layer,
        )


class RealAmplitudes(TwoLocal):
    """Layer Template for a RealAmplitudes circuit

    Args:
        arch (dict): circuit architecture in a dictionary format
        entanglement_layer (str): type of entanglement layer in a string ("linear", "reverse_linear", "circular", "full") or tq.QuantumModule format
        reps (int): number of reptitions of the rotation and entanglement layers in a integer format
        skip_final_rotation_layer (bool): whether or not to add the final rotation layer as a boolean
    """

    def __init__(
        self,
        arch: dict = None,
        entanglement_layer: str = "reverse_linear",
        reps: int = 3,
        skip_final_rotation_layer: bool = False,
    ):
        # construct circuit with rotation layers of RY and entanglement with CX
        super().__init__(
            arch=arch,
            rotation_ops=[tq.RY],
            entanglement_ops=[tq.CNOT],
            entanglement_layer=entanglement_layer,
            reps=reps,
            skip_final_rotation_layer=skip_final_rotation_layer,
        )


class PauliTwoDesign(TwoLocal):
    """Layer Template for a PauliTwoDesign circuit

    Args:
        arch (dict): circuit architecture in a dictionary format
        entanglement_layer (str): type of entanglement layer in a string ("linear", "reverse_linear", "circular", "full") or tq.QuantumModule format
        reps (int): number of reptitions of the rotation and entanglement layers in a integer format
        skip_final_rotation_layer (bool): whether or not to add the final rotation layer as a boolean
    """

    def __init__(
        self,
        arch: dict = None,
        entanglement_layer: str = "reverse_linear",
        reps: int = 3,
        skip_final_rotation_layer: bool = False,
        seed: int = 0,
    ):
        # set seed
        self.seed = seed
        # construct circuit with entanglement with CX
        super().__init__(
            arch=arch,
            entanglement_ops=[tq.CNOT],
            entanglement_layer=entanglement_layer,
            reps=reps,
            skip_final_rotation_layer=skip_final_rotation_layer,
        )

    def build_initial_layer(self):
        # add an initial layer of ry with rotation pi/4
        return tq.QuantumModule.from_op_history(
            [
                {"name": "ry", "wires": wire, "params": torch.pi / 4}
                for wire in range(self.arch["n_wires"])
            ]
        )

    def build_rotation_block(self):
        # make a random layer of rotations
        return [RandomOp1All(n_wires=self.n_wires, seed=self.seed)]
