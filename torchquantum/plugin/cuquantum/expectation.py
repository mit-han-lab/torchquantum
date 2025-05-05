# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: MIT

from typing import List, Dict, Union

import torch.nn as nn

from .utils import check_input_params
from .backend import QuantumBackend
from .circuit import ParameterizedQuantumCircuit


class QuantumExpectation(nn.Module):
    """A PyTorch module for computing expectation values of Pauli operators.

    This module computes the expectation values of specified Pauli operators
    in the quantum state prepared by a given quantum circuit.

    Args:
        circuit: The quantum circuit that prepares the state.
        backend: The quantum backend to use for computation.
        pauli_ops: List of Pauli operators to compute expectations for. Each Pauli operator can be either:
            - A single Pauli string specifying the pauli operator for each qubit ("I", "X", "Y", or "Z").
            - A linear combination of Pauli strings specified as a dictionary mapping each single Pauli string to
              its corresponding coefficient.
    """

    def __init__(
        self,
        circuit: ParameterizedQuantumCircuit,
        backend: QuantumBackend,
        pauli_ops: Union[List[str], Dict[str, float]],
    ):
        super().__init__()
        self._circuit = circuit.copy()
        self._pauli_ops = pauli_ops.copy()
        self._backend = backend
        self._expectation_module = self.backend._create_expectation_module(circuit, pauli_ops)

    def forward(self, input_params=None):
        """Compute the expectation values for the Pauli operators specified in the constructor.

        Args:
            input_params: 2D Tensor of input parameters for the circuit. Shape should be (batch_size, n_input_params). If
                only one batch is being processed, the tensor can be instead a 1D tensor with shape (n_input_params,). If
                the circuit has no input parameters, this argument can be omitted (i.e. None).

        Returns:
            2D Tensor of expectation values for each Pauli operator in each batch. The shape is (batch_size, len(pauli_ops)).
        """
        input_params = check_input_params(input_params, self._circuit.n_input_params)
        return self._expectation_module(input_params)

    @property
    def pauli_ops(self):
        """Get the list of Pauli operators being measured."""
        return self._pauli_ops.copy()

    @property
    def circuit(self):
        """Get the quantum circuit used for state preparation."""
        return self._circuit.copy()

    @property
    def backend(self):
        """Get the quantum backend being used for computation."""
        return self._backend
