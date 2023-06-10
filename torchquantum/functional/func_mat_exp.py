import torch
from .functional import gate_wrapper
from typing import Union
import numpy as np

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
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=qdev,
        wires=wires,
        n_wires=n_wires,
        inverse=inverse,
    )
