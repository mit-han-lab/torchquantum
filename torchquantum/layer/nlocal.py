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
import torchquantum.layers as layers


class NLocal(layers.LayerTemplate0):
    """Layer Template for a NLocal Class

    Args:
        rotation_ops (list): gates for the rotation layer as a list of torchquantum operations
        entanglement_ops (list): gates for the entanglement layer as a list of torchquantum operations
        arch (dict): circuit architecture in a dictionary format
        rotation_layer (torchquantum.QuantumModule): type of rotation layer in a torchquantum.QuantumModule format
        entanglement_layer (torchquantum.QuantumModule): type of entanglement layer in a torchquantum.QuantumModule format
        rotation_layer_params (dict): additional parameters for the rotation layer in a dictionary format
        entanglement_layer_params (dict): additional parameters for the entanglement layer in a dictionary format
        skip_final_rotation_layer (bool): whether or not to add the final rotation layer as a boolean
    """

    def __init__(
        self,
        rotation_ops: list,
        entanglement_ops: list,
        arch: dict = None,
        rotation_layer: tq.QuantumModule = tq.layers.Op1QAllLayer,
        entanglement_layer: tq.QuantumModule = tq.layers.Op2QAllLayer,
        rotation_layer_params: dict = {},
        entanglement_layer_params: dict = {},
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
        self.skip_final_rotation_layer = skip_final_rotation_layer
        layers.LayerTemplate0.__init__(self, arch)

    def build_layers(self):
        layers_all = tq.QuantumModuleList()

        # repeat for each rep
        for _ in range(self.n_blocks):
            # add rotation blocks to the qubits
            for rot in self.rotation_ops:
                layers_all.append(
                    self.rotation_layer(
                        op=rot, n_wires=self.n_wires, *self.rotation_layer_params
                    )
                )

            # add entanglement blocks to everything
            for entanglement in self.entanglement_ops:
                layers_all.append(
                    self.entanglement_layer(
                        op=entanglement,
                        n_wires=self.n_wires,
                        *self.entanglement_layer_params
                    )
                )

        # add final rotation layer
        if not self.skip_final_rotation_layer:
            for rot in self.rotation_ops:
                layers_all.append(
                    self.rotation_layer(
                        op=rot, n_wires=self.n_wires, *self.rotation_layer_params
                    )
                )

        # return QuantumModuleList
        return layers_all
