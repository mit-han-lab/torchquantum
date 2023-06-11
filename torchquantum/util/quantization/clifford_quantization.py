"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torchquantum as tq
import numpy as np

from typing import Any


class QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> Any:
        # should be round so that the changes would be small, values close to
        # 2pi should go to 2pi
        return x.round()

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        grad_input = grad_output.clone()
        mean, std = grad_input.mean(), grad_input.std()
        return grad_input.clamp_(mean - 3 * std, mean + 3 * std)


class CliffordQuantizer(object):
    def __init__(self):
        pass

    # straight-through estimator
    @staticmethod
    def quantize_sse(params):
        param = params[0][0]
        param = param % (2 * np.pi)
        param = np.pi / 2 * QuantizeFunction.apply(param / (np.pi / 2))
        params = param.unsqueeze(0).unsqueeze(0)
        return params
