# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: MIT

from typing import List, Union, Dict, Optional

from torch import nn
from cuquantum.tensornet.experimental import TNConfig, MPSConfig

from ..backend import QuantumBackend
from ..circuit import ParameterizedQuantumCircuit
from .state import ParameterizedNetworkState
from .expectation import CuTNExpectationFD
from .amplitude import CuTNAmplitudeFD
from .sampling import CuTNSampling


class CuTensorNetworkBackend(QuantumBackend):
    """A backend implementation using cuQuantum's Tensor Network library for quantum circuit simulations.

    This backend provides functionality for computing expectation values, amplitudes, and sampling from quantum circuits using
    tensor network methods. It supports both general tensor networks and Matrix Product States (MPS).

    Args:
        config: Optional configuration for the tensor network simulation. Can be either a
            :py:class:`TNConfig <cuquantum.tensornet.experimental.TNConfig>` or
            :py:class:`MPSConfig <cuquantum.tensornet.experimental.MPSConfig>` object.
        allow_multiple_states: If False, the backend uses a single network state for each quantum PyTorch module.
            If True, the backend may create separate network states to utilize caching when necessary.
            This is e.g. useful when the same quantum circuit is used to compute expectation values of different Pauli
            operators. This can speed up the computation at the cost of slightly increased memory usage (one network state
            per Pauli operator). Default is True.
        grad_method: Method for computing gradients. Currently only supports "finite_difference".
        fd_delta: Step size for finite difference gradient computation.
    """

    def __init__(
        self,
        config=Optional[Union[TNConfig, MPSConfig]],
        allow_multiple_states: bool = True,
        grad_method: str = "finite_difference",
        fd_delta: float = 1e-4,
    ):
        self._allow_multiple_states = allow_multiple_states
        self._config = config
        self._grad_method = grad_method
        self._fd_delta = fd_delta
        if not self._grad_method in ["finite_difference"]:
            raise NotImplementedError(f"Unkown gradient method")

    def _create_expectation_module(
        self, circuit: ParameterizedQuantumCircuit, pauli_ops: Union[List[str], Dict[str, float]]
    ) -> nn.Module:
        if self._allow_multiple_states:
            # In order to utilize caching feature of the network states, we need to create a seperate network state for each pauli operator.
            # Otherwise, the network state cache will be overwritten when pauli_op changes.
            states = [
                ParameterizedNetworkState.from_parameterized_circuit(circuit, self._config)
                for _ in range(len(pauli_ops))
            ]
        else:
            states = [ParameterizedNetworkState.from_parameterized_circuit(circuit, self._config)] * len(pauli_ops)

        if self._grad_method == "finite_difference":
            return CuTNExpectationFD(states, pauli_ops, circuit.trainable_params, self._fd_delta)
        else:
            raise NotImplementedError(f"Gradient method {self._grad_method} not supported for this backend")

    def _create_amplitude_module(self, circuit: ParameterizedQuantumCircuit, bitstrings: List[str]) -> nn.Module:
        state = ParameterizedNetworkState.from_parameterized_circuit(circuit, self._config)
        if self._grad_method == "finite_difference":
            return CuTNAmplitudeFD(state, bitstrings, circuit.trainable_params, self._fd_delta)
        else:
            raise NotImplementedError(f"Gradient method {self._grad_method} not supported for this backend")

    def _create_sampling_module(
        self, circuit: ParameterizedQuantumCircuit, n_samples: int, wires: Optional[List[int]] = None
    ):
        state = ParameterizedNetworkState.from_parameterized_circuit(circuit, self._config)
        return CuTNSampling(state, n_samples, wires, circuit.trainable_params)
