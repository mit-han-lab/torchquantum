# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from collections import namedtuple
from typing import List, Optional

import torch
import torch.nn as nn
from torchquantum.operator import Operator
from torchquantum.operator.op_types import AnyNParams, AnyWires
from torchquantum.operator.standard_gates import all_variables
from torchquantum.operator.standard_gates.reset import Reset


class _ParameterizedQuantumGate:
    """A named tuple representing a parameterized quantum gate in a circuit.

    This class holds the information needed to represent a quantum gate with parameters
    that can be either trainable, input parameters, or fixed values.

    Attributes:
        matrix_generator: Function that generates the gate's unitary matrix given parameters as an argument.
        wires: List of qubit indices the gate acts on
        params: Current parameter values for the gate
        trainable_idx: Indices of parameters that are trainable
        input_idx: Indices of parameters that are input parameters
        inverse: Whether the gate should be applied in inverse
    """


_ParameterizedQuantumGate = namedtuple(
    "Gate", ["matrix_generator", "wires", "params", "trainable_idx", "input_idx", "inverse"]
)


class ParameterizedQuantumCircuit:
    """A class representing a parameterized quantum circuit.

    This class allows building quantum circuits with both trainable and input parameters.
    Gates can be added to the circuit with parameters that are either trainable,
    input parameters, or fixed values.

    Args:
        n_wires: Number of qubits in the circuit
        n_input_params: Number of input parameters the circuit accepts
        n_trainable_params: Number of trainable parameters in the circuit
    """

    def __init__(self, n_wires: int, n_input_params: int = 0, n_trainable_params: int = 0):
        super().__init__()
        self._n_wires = n_wires
        self._n_input_params = n_input_params
        self._n_trainable_params = n_trainable_params
        self._gates = []
        self._trainable_params = nn.Parameter(torch.zeros(n_trainable_params))

    @property
    def n_wires(self):
        """Get the number of qubits in the circuit."""
        return self._n_wires

    @property
    def n_input_params(self):
        """Get the number of input parameters the circuit accepts."""
        return self._n_input_params

    @property
    def n_trainable_params(self):
        """Get the number of trainable parameters in the circuit."""
        return self._n_trainable_params

    @property
    def gates(self):
        """Get the list of gates in the circuit."""
        return self._gates

    @property
    def trainable_params(self):
        """Get the trainable parameters of the circuit."""
        return self._trainable_params

    def copy(self):
        """Creates a shallow copy of the circuit.

        The parameters are shared, but appending new gates will not affect the original circuit.

        Returns:
            A new ParameterizedQuantumCircuit instance with the same gates and parameters
        """
        circuit = ParameterizedQuantumCircuit(self._n_wires, self._n_input_params, self._n_trainable_params)
        circuit._trainable_params = self._trainable_params
        circuit._gates = self._gates[:]
        return circuit

    def append_gate(
        self,
        op: Operator,
        wires: List[int],
        fixed_params: Optional[List[float]] = None,
        trainable_idx: Optional[List[int]] = None,
        input_idx: Optional[List[int]] = None,
        inverse: bool = False,
    ):
        """Add a gate to the circuit.

        Args:
            op: The quantum operator to apply. It can be any of the TorchQuantum operators defined in
                :py:mod:`torchquantum.operator.standard_gates` with a fixed number of parameters except for
                :py:class:`Reset <torchquantum.operator.standard_gates.reset.Reset>`. Note that 
            wires: List of qubit(s) to apply the gate to.
            fixed_params: List of numbers defining the values of the fixed parameters for the gate. The length of this 
                list must be the same as the number of parameters for the gate. Gate parameters that are not fixed 
                should be set to None in this list. If the gate has no fixed parameters, this argument can be omitted 
                (i.e. None).
            trainable_idx: List of indices linking the gate parameters to the circuit's trainable parameters. The length
                of this list must be the same as the number of parameters for the gate. Gate parameters that are not
                trainable should be set to None in this list. If the gate has no trainable parameters, this argument can
                be omitted (i.e. None).
            input_idx: List of indices linking the gate parameters to the circuit's input parameters. The length of this
                list must be the same as the number of parameters for the gate. Gate parameters that are not input
                parameters should be set to None in this list. If the gate has no input parameters, this argument can be
                omitted (i.e. None).
            inverse: Whether to apply the inverse of the operator

        Raises:
            ValueError: If the operator is invalid, wires are out of bounds, or parameter indices are invalid.
        """
        if op not in all_variables:
            raise ValueError(f"{op} is not a valid operator")

        if isinstance(op, Reset):
            raise ValueError(f"{op} is not supported")

        if op.num_params == AnyNParams:
            raise ValueError(f"{op} has a variable number of parameters. This is not supported yet.")

        name = op.__name__
        if isinstance(wires, int):
            wires = [wires]
        if op.num_wires != AnyWires and len(wires) != op.num_wires:
            raise ValueError(f"Number of wires for {name} must be {op.num_wires}")
        for wire in wires:
            if wire < 0 or wire >= self._n_wires:
                raise ValueError(f"Wire {wire} is out of bounds")

        n_params = op.num_params

        if fixed_params is None:
            fixed_params = [None] * n_params
        if isinstance(fixed_params, float):
            fixed_params = [fixed_params]
        if not isinstance(fixed_params, list) or len(fixed_params) != n_params:
            raise ValueError(f"Fixed params must be a list of floats/None of length {n_params}")

            
        if trainable_idx is None:
            trainable_idx = [None] * n_params
        if isinstance(trainable_idx, int):
            trainable_idx = [trainable_idx]
        if not isinstance(trainable_idx, list) or len(trainable_idx) != n_params:
            raise ValueError(f"Trainable index must be an integer or a list of integers/None of length {n_params}")
        for idx in trainable_idx:
            if idx is not None and (idx < 0 or idx >= self._n_trainable_params):
                raise ValueError(f"Trainable index {idx} is out of bounds")

        if input_idx is None:
            input_idx = [None] * n_params
        if isinstance(input_idx, int):
            input_idx = [input_idx]
        if not isinstance(input_idx, list) or len(input_idx) != n_params:
            raise ValueError(f"Input index must be an integer or a list of integers/None of length {n_params}")
        for idx in input_idx:
            if idx is not None and (idx < 0 or idx >= self._n_input_params):
                raise ValueError(f"Input index {idx} is out of bounds")

        params = torch.empty(op.num_params)
        for p in range(n_params):
            if fixed_params[p] is not None:
                if(trainable_idx[p] is not None):
                    raise ValueError(f"Parameter {p} cannot be both fixed and trainable")
                if(input_idx[p] is not None):
                    raise ValueError(f"Parameter {p} cannot be both fixed and an input")
                params[p] = fixed_params[p]
            else:
                if trainable_idx[p] is not None and input_idx[p] is not None:
                    raise ValueError(f"Parameter {p} cannot be both trainable and an input")
                if trainable_idx[p] is None and input_idx[p] is None:
                    raise ValueError(f"Parameter {p} must be either fixed, trainable, or an input")
                
        matrix_generator = _maxtrix_generator_from_operator(op, len(wires))

        self._gates.append(
            _ParameterizedQuantumGate(matrix_generator, wires, params, trainable_idx, input_idx, inverse)
        )

    def set_trainable_params(self, trainable_params: torch.Tensor):
        """Set the trainable parameters of the circuit.

        Args:
            trainable_params: A tensor of trainable parameters
        """
        with torch.no_grad():
            for i in range(self._n_trainable_params):
                self._trainable_params[i] = trainable_params[i]


def _maxtrix_generator_from_operator(op, n_wires):
    if op.num_wires == AnyWires: # This is necessary for operators that act on any number of wires, e.g. QFT, MultiCNOT, MultiRZ, etc.
        return lambda params: op._matrix(params.unsqueeze(0), n_wires).reshape((2,) * (2 * n_wires))
    else:
        return lambda params: op._matrix(params.unsqueeze(0)).reshape((2,) * (2 * n_wires))
