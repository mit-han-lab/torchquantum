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
from torchquantum.module import QuantumModule
from torchquantum.algorithm import Hamiltonian
import torchquantum.functional as tqf


__all__ = ['OpHamilExp']


class OpHamilExp(QuantumModule):
    """Matrix exponential operation.
    exp(-i * theta * H / 2)
    the default theta is 0.0
    """
    def __init__(self,
                 hamil: Hamiltonian,
                 trainable: bool = True,
                 theta: float = 0.0):
        """Initialize the OpHamilExp module.
        
        Args:
            hamil: The Hamiltonian.
            has_params: Whether the module has parameters.
            trainable: Whether the parameters are trainable.
            theta: The initial value of theta.
        
        """
        super().__init__()
        if trainable:
            self.theta = torch.nn.parameter.Parameter(torch.tensor(theta))
        else:
            self.theta = torch.tensor(theta)
        self.hamil = hamil
    
    def get_exponent_matrix(self):
        """Get the matrix on exponent."""
        return self.hamil.matrix * -1j * self.theta / 2
    
    @property
    def exponent_matrix(self):
        """Get the matrix on exponent."""
        return self.get_exponent_matrix()

    def get_matrix(self):
        """Get the overall matrix."""
        return torch.matrix_exp(self.exponent_matrix)
    
    @property
    def matrix(self):
        """Get the overall matrix."""
        return self.get_matrix()
    
    def forward(self, qdev, wires):
        """Forward the OpHamilExp module.
        Args:
            qdev: The QuantumDevice.
            wires: The wires.

        """
        matrix = self.matrix.to(qdev.device)
        tqf.qubitunitaryfast(
            q_device=qdev,
            wires=wires,
            params=matrix,
        )
