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
from torchquantum.layer.layers import RandomOp1All
from .two_local import TwoLocal

__all__ = [
    "PauliTwoDesign",
]


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
        n_wires: int = 1,
        entanglement_layer: str = "reverse_linear",
        reps: int = 3,
        skip_final_rotation_layer: bool = False,
        seed: int = 0,
    ):
        # set seed
        self.seed = seed
        # construct circuit with entanglement with CX
        super().__init__(
            n_wires=n_wires,
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
