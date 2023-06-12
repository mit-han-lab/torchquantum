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
from .functionals import gate_wrapper
from typing import Union
import numpy as np
import torchquantum.functional as tqf

__all__ = ["matrix_exp"]

def matrix_exp(
    qdev,
    wires,
    params: Union[torch.Tensor, np.ndarray],
    n_wires=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the matrix exponential operation.

    Args:
        qdev (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    if isinstance(params, np.ndarray):
        params = torch.from_numpy(params)

    mat = torch.matrix_exp(params)

    name = 'qubitunitaryfast'

    tqf.qubitunitaryfast(
        q_device=qdev,
        wires=wires,
        params=mat,
    )
