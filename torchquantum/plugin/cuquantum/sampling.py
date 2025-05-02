# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: MIT

from typing import List, Optional

import torch.nn as nn

from .utils import check_input_params
from .backend import QuantumBackend
from .circuit import ParameterizedQuantumCircuit


class QuantumSampling(nn.Module):
    """A PyTorch module for sampling from quantum states.

    This module generates samples from the quantum state prepared by a given quantum circuit. It can sample from all 
    qubits or a specified subset of qubits.

    Args:
        circuit: The quantum circuit that prepares the state.
        n_samples: Number of samples to generate per batch.
        backend: The quantum backend to use for computation.
        wires: Optional list of wires/qubits to sample from. If not provided, all wires/qubits are sampled from.
    """
    
    def __init__(self, circuit:ParameterizedQuantumCircuit, n_samples: int, backend: QuantumBackend, wires: Optional[List[int]]=None):
        super().__init__()
        self.circuit = circuit
        self.n_samples = n_samples
        self.wires = wires
        self.backend = backend
        self.sampling_module = self.backend._create_sampling_module(circuit, n_samples, wires)

    def forward(self, input_params=None):
        """Generate samples from the quantum state.

        Args:
            input_params: 2D Tensor of input parameters for the circuit. Shape should be (batch_size, n_input_params). If
                only one batch is being processed, the tensor can be instead a 1D tensor with shape (n_input_params,). If
                the circuit has no input parameters, this argument can be omitted (i.e. None).

        Returns:
            List of samples with length batch_size. Each sample is a dictionary mapping the bitstring to the corresponding
            count.
        """
        input_params = check_input_params(input_params, self.circuit.n_input_params)
        return self.sampling_module(input_params)
