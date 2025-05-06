# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch
from torch import nn
from cuquantum.tensornet.experimental import NetworkOperator

from .gradient import CuTNFiniteDifference


class CuTNExpectationFD(nn.Module):
    def __init__(self, states, pauli_ops, circuit_params, delta):
        super().__init__()
        if len(states) != len(pauli_ops):
            raise ValueError(f"Expected as many states as pauli operators, got {len(states)} and {len(pauli_ops)}")
        if len(states) == 0:
            raise ValueError(f"Expected at least one state")

        self.n_exp_vals = len(pauli_ops)
        self.states = states
        self.pauli_ops = []
        self.output_dtype = torch.float32
        for i in range(self.n_exp_vals):
            self.pauli_ops.append(NetworkOperator.from_pauli_strings(pauli_ops[i], dtype=states[i].dtype))
            if states[i].dtype == "float64" or states[i].dtype == "complex128":
                self.output_dtype = torch.float64
            elif states[i].dtype == "float32" or states[i].dtype == "complex64":
                pass
            else:
                raise ValueError(f"Unkown state dtype: {states[i].dtype}")

        self.delta = delta
        self.circuit_params = circuit_params

    def forward(self, input_params):
        exp_vals = torch.zeros(input_params.shape[0], self.n_exp_vals, dtype=self.output_dtype)
        for batch_idx in range(input_params.shape[0]):
            for exp_val_idx in range(self.n_exp_vals):
                exp_vals[batch_idx, exp_val_idx] = CuTNFiniteDifference.apply(
                    self.states[exp_val_idx],
                    _expectation_wrapper,
                    self.pauli_ops[exp_val_idx],
                    self.delta,
                    self.circuit_params,
                    input_params[batch_idx],
                )
        return exp_vals


def _expectation_wrapper(state, operator):
    value = state.compute_expectation(operator)

    if state.dtype == "float32" or state.dtype == "complex64":
        if abs(value.imag) > 1e-6:
            raise RuntimeWarning(f"Something is wrong. Expectation value is not real. Value: {value}")
    elif state.dtype == "float64" or state.dtype == "complex128":
        if abs(value.imag) > 1e-15:
            raise RuntimeWarning(f"Something is wrong. Expectation value is not real. Value: {value}")
    else:
        raise ValueError(f"Unknown dtype: {state.dtype}")

    return value.real
