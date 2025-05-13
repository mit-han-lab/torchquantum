# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from collections import defaultdict

import torch
from torchquantum.macro import C_DTYPE
from cuquantum.tensornet.experimental import NetworkState


class ParameterizedTensorOperator:
    def __init__(self, modes, tensor_generator, params, parameters_map, unitary, adjoint):
        self.modes = modes
        self.tensor_generator = tensor_generator
        self.params = params
        self.parameters_map = parameters_map
        self.unitary = unitary
        self.adjoint = adjoint

    @classmethod
    def from_gate(cls, gate, trainable_args_idx=0, input_args_idx=1):
        parameters_map = {}

        for param_idx in range(len(gate.params)):
            if gate.trainable_idx[param_idx] is not None:
                parameters_map[param_idx] = (trainable_args_idx, gate.trainable_idx[param_idx])
            if gate.input_idx[param_idx] is not None:
                parameters_map[param_idx] = (input_args_idx, gate.input_idx[param_idx])

        return cls(gate.wires, gate.matrix_generator, gate.params, parameters_map, True, gate.inverse)

    def update(self, network_state, tensor_id, *args):
        for param_idx, (arg_idx, val_idx) in self.parameters_map.items():
            self.params[param_idx] = args[arg_idx][val_idx]

        tensor = self.tensor_generator(self.params)
        network_state.update_tensor_operator(tensor_id, tensor, unitary=self.unitary)


class ParameterizedNetworkState(NetworkState):
    """
    A NetworkState that can be parameterized.
    """

    def __init__(self, param_args_shapes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_args_shapes = param_args_shapes
        self.mutable_operators = {}  # tensor_id -> operator
        self.reverse_params_map = defaultdict(set)  # (arg_idx, val_idx) -> set of tensor_ids

    def apply_parameterized_tensor_operator(self, operator: ParameterizedTensorOperator):
        operand = operator.tensor_generator(operator.params)
        immutable = not operator.parameters_map
        tensor_id = super().apply_tensor_operator(
            operator.modes, operand, immutable=immutable, unitary=operator.unitary, adjoint=operator.adjoint
        )
        if not immutable:
            self.mutable_operators[tensor_id] = operator
            for arg_idx, val_idx in operator.parameters_map.values():
                self.reverse_params_map[(arg_idx, val_idx)].add(tensor_id)
        return tensor_id

    def update_all_parameters(self, *args):
        if len(args) != len(self.param_args_shapes):
            raise ValueError(f"Expected {len(self.param_args_shapes)} arguments, got {len(args)}")
        for arg_idx, arg_shape in enumerate(self.param_args_shapes):
            if args[arg_idx].ndim != 1:
                raise ValueError(f"Expected argument {arg_idx} to be a 1D tensor, got {args[arg_idx].ndim}D tensor")
            if args[arg_idx].size(0) != arg_shape:
                raise ValueError(f"Expected argument {arg_idx} to have shape {arg_shape}, got {args[arg_idx].size(0)}")

        for tensor_id, operator in self.mutable_operators.items():
            operator.update(self, tensor_id, *args)

    def update_parameter(self, arg_idx, val_idx, *args):
        for tensor_id in self.reverse_params_map[(arg_idx, val_idx)]:
            self.mutable_operators[tensor_id].update(self, tensor_id, *args)

    @classmethod
    def from_parameterized_circuit(cls, circuit, config):
        if C_DTYPE == torch.complex64:
            dtype = "complex64"
        elif C_DTYPE == torch.complex128:
            dtype = "complex128"
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        state = cls(
            param_args_shapes=[circuit.n_trainable_params, circuit.n_input_params],
            state_mode_extents=(2,) * circuit.n_wires,
            dtype=dtype,
            config=config,
        )
        for gate in circuit._gates:
            operator = ParameterizedTensorOperator.from_gate(gate, 0, 1)
            state.apply_parameterized_tensor_operator(operator)

        return state
