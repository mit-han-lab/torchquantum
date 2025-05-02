# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: MIT

import torch
from torch import nn

from .state import ParameterizedNetworkState
from .gradient import CuTNFiniteDifference


class CuTNAmplitudeFD(nn.Module):
    def __init__(self, state, bitstrings, circuit_params, delta):
        super().__init__()

        self.n_amplitudes = len(bitstrings)
        self.state = state
        self.bitstrings = bitstrings
        if state.dtype == "float64" or state.dtype == "complex128":
            self.output_dtype = torch.complex128
        elif state.dtype == "float32" or state.dtype == "complex64":
            self.output_dtype = torch.complex64
        else:
            raise ValueError(f"Unkown state dtype: {state.dtype}")
        self.delta = delta
        self.circuit_params = circuit_params

    def forward(self, input_params):
        amplitudes = torch.zeros(input_params.shape[0], self.n_amplitudes, dtype=self.output_dtype)
        for batch_idx in range(input_params.shape[0]):
            for amplitude_idx in range(self.n_amplitudes):
                amplitudes[batch_idx, amplitude_idx] = CuTNFiniteDifference.apply(
                    self.state,
                    _amplitude_wrapper,
                    self.bitstrings[amplitude_idx],
                    self.delta,
                    self.circuit_params,
                    input_params[batch_idx],
                )
        return amplitudes


def _amplitude_wrapper(state: ParameterizedNetworkState, bitstring: str):
    return state.compute_amplitude(bitstring)
