# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import List

import torch.nn as nn

from .utils import check_input_params
from .backend import QuantumBackend
from .circuit import ParameterizedQuantumCircuit


class QuantumAmplitude(nn.Module):
    """A PyTorch module for computing quantum state amplitudes.

    This module computes the amplitudes of specified bitstrings in the quantum state prepared by a given quantum circuit.

    Args:
        circuit: The quantum circuit that prepares the state.
        backend: The quantum backend to use for computation.
        bitstrings: List of bitstrings whose amplitudes to compute.
    """

    def __init__(self, circuit: ParameterizedQuantumCircuit, backend: QuantumBackend, bitstrings: List[str]):
        super().__init__()
        self._circuit = circuit.copy()
        self._bitstrings = bitstrings.copy()
        self._backend = backend
        self._amplitude_module = self.backend._create_amplitude_module(circuit, bitstrings)

    def forward(self, input_params=None):
        """Compute the amplitudes for the bitstrings specified in the constructor.

        Args:
            input_params: 2D Tensor of input parameters for the circuit. Shape should be (batch_size, n_input_params). If
                only one batch is being processed, the tensor can be instead a 1D tensor with shape (n_input_params,). If
                the circuit has no input parameters, this argument can be omitted (i.e. None).

        Returns:
            2D Tensor of amplitudes for each bitstring in each batch. The shape is (batch_size, len(bitstrings)).
        """
        input_params = check_input_params(input_params, self._circuit.n_input_params)
        return self._amplitude_module(input_params)

    @property
    def bitstrings(self):
        """Get the list of bitstrings whose amplitudes are being computed."""
        return self._bitstrings.copy()

    @property
    def circuit(self):
        """Get the quantum circuit used for state preparation."""
        return self._circuit.copy()

    @property
    def backend(self):
        """Get the quantum backend being used for computation."""
        return self._backend
