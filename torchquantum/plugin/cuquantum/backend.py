# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional

import torch.nn as nn

from .circuit import ParameterizedQuantumCircuit


class QuantumBackend(ABC):
    """Abstract base class for quantum backends.

    This class defines the interface that all quantum backends must implement. Each backend must provide methods for 
    creating PyTorch modules that compute:
    - Expectation values of Pauli operators.
    - State amplitudes for given bitstrings.
    - Sampling from the quantum state.
    """

    @abstractmethod
    def _create_expectation_module(
        self, circuit: ParameterizedQuantumCircuit, pauli_ops: Union[List[str], Dict[str, float]]
    ) -> nn.Module:
        """Create a module for computing expectation values of Pauli operators.

        Args:
            circuit: The quantum circuit that prepares the state
            pauli_ops: List of Pauli operators to compute expectations for. Each Pauli operator can be either:
                - A single Pauli string specifying the pauli operator for each qubit ("I", "X", "Y", or "Z").
                - A linear combination of Pauli strings specified as a dictionary mapping each single Pauli string to its 
                  corresponding coefficient.

        Returns:
            A PyTorch module that computes the expectation values.
        """
        pass

    @abstractmethod
    def _create_amplitude_module(self, circuit: ParameterizedQuantumCircuit, bitstrings: List[str]) -> nn.Module:
        """Create a module for computing state amplitudes.

        Args:
            circuit: The quantum circuit that prepares the state.
            bitstrings: List of bitstrings whose amplitudes to compute.

        Returns:
            A PyTorch module that computes the amplitudes.
        """
        pass

    @abstractmethod
    def _create_sampling_module(
        self, circuit: ParameterizedQuantumCircuit, n_samples: int, wires: Optional[List[int]] = None
    ) -> nn.Module:
        """Create a module for sampling from the quantum state.

        Args:
            circuit: The quantum circuit that prepares the state.
            n_samples: Number of samples to generate.
            wires: Optional list of wires/qubits to sample from. If not provided, all wires/qubits are sampled from.

        Returns:
            A PyTorch module that generates samples from the quantum state.
        """
        pass
