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

import torchquantum as tq
from torchquantum.layer.layers import (
    Op1QAllLayer,
)
from torchquantum.layer.entanglement import (
    Op2QAllLayer,
    Op2QDenseLayer,
)
from torchquantum.operator import op_name_dict
from .nlocal import NLocal
from collections.abc import Iterable

__all__ = [
    "TwoLocal",
]


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
        n_wires: int = 1,
        rotation_ops: list = None,
        entanglement_ops: list = None,
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

        # handling different input types for the rotation ops
        if isinstance(rotation_ops, str):
            rotation_ops = [op_name_dict[rotation_ops]]
        elif isinstance(rotation_ops, Iterable):
            if all(isinstance(rot, str) for rot in rotation_ops):
                rotation_ops = [op_name_dict[rot] for rot in rotation_ops]
        else:
            rotation_ops = [rotation_ops]

        # handling different input types for the entanglment ops
        if isinstance(entanglement_ops, str):
            entanglement_ops = [op_name_dict[entanglement_ops]]
        elif isinstance(entanglement_ops, Iterable):
            if all(isinstance(op, str) for op in entanglement_ops):
                entanglement_ops = [op_name_dict[op] for op in entanglement_ops]
        else:
            entanglement_ops = [entanglement_ops]

        # initialize
        super().__init__(
            arch={"n_wires": n_wires},
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
