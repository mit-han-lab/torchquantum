# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch

def check_input_params(input_params, n_params):
    """Validate and format input parameters for quantum circuits.

    This function ensures that input parameters are properly formatted as a 2D tensor with the correct number of parameters 
    per batch.

    Args:
        input_params: Input parameters tensor. Can be None, 1D, or 2D.
        n_params: Expected number of parameters per batch.

    Returns:
        A 2D tensor of shape (batch_size, n_params) containing the input parameters.

    Raises:
        ValueError: If input_params is not a 1D or 2D tensor, or if it has the wrong number of parameters per batch.
    """
    if(input_params is None):
        input_params = torch.zeros(0, dtype=torch.float32)
    if(input_params.ndim == 1): # no batching, make it a batch of size 1
        input_params = input_params.unsqueeze(0)
    if(input_params.ndim != 2):
        raise ValueError(f"Input must be a 1D or 2D tensor")
    
    if(input_params.shape[1] != n_params):
        raise ValueError(f"Input must have {n_params} parameters per batch")
    
    return input_params