# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: MIT

import torch


class CuTNFiniteDifference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, operation, operation_argument, delta: float, *args):
        ctx.save_for_backward(*[arg.detach().clone() for arg in args])  # Save tensors for backward
        ctx.state = state
        ctx.operation = operation
        ctx.operation_argument = operation_argument
        ctx.delta = delta

        state.update_all_parameters(*args)

        return torch.tensor(operation(state, operation_argument))

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: compute gradients"""
        args = ctx.saved_tensors
        state = ctx.state
        operation = ctx.operation
        operation_argument = ctx.operation_argument
        delta = ctx.delta

        # restore all original parameters
        state.update_all_parameters(*args)

        grads = [None] * len(args)

        for arg_idx, arg in enumerate(args):
            if ctx.needs_input_grad[4 + arg_idx]:
                grads[arg_idx] = torch.zeros_like(arg)
                for var_idx in range(grads[arg_idx].shape[0]):
                    original_arg_val = arg[var_idx].item()
                    arg[var_idx] = original_arg_val - delta / 2
                    state.update_parameter(arg_idx, var_idx, *args)
                    val_minus = operation(state, operation_argument)

                    arg[var_idx] = original_arg_val + delta / 2
                    state.update_parameter(arg_idx, var_idx, *args)
                    val_plus = operation(state, operation_argument)

                    grads[arg_idx][var_idx] = grad_output * (val_plus - val_minus) / delta

                    arg[var_idx] = original_arg_val
                    state.update_parameter(arg_idx, var_idx, *args)

        return None, None, None, None, *grads
