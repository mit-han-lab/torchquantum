# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch.nn as nn


class CuTNSampling(nn.Module):
    def __init__(self, state, n_samples, wires, circuit_params):
        super().__init__()
        self.state = state
        self.n_samples = n_samples
        self.wires = wires
        self.circuit_params = circuit_params

    def forward(self, input_params):
        samples = []
        for batch_idx in range(input_params.shape[0]):
            self.state.update_all_parameters(self.circuit_params, input_params[batch_idx])
            samples.append(self.state.compute_sampling(self.n_samples, modes=self.wires))

        return samples
