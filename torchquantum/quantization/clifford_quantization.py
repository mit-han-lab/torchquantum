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
        param = np.pi / 2 * QuantizeFunction.apply(param /
                                                   (np.pi / 2))
        params = param.unsqueeze(0).unsqueeze(0)
        return params
