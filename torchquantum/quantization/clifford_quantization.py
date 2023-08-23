import torch
import torchquantum as tq
import numpy as np

from typing import Any


class QuantizeFunction(torch.autograd.Function):
    """A custom quantization function for use in the backward pass of CliffordQuantizer.

    Methods:
        forward(ctx, x): Forward pass of the quantization function.

        backward(ctx, grad_output): Backward pass of the quantization function.
    """
    
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> Any:
        """
        Forward pass of the quantization function.

        Args:
            ctx (Any): Context object.
            x (torch.Tensor): Input tensor.

        Returns:
            Any: Quantized output tensor.
        """
        
        # should be round so that the changes would be small, values close to
        # 2pi should go to 2pi
        return x.round()

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        """
        Backward pass of the quantization function.

        Args:
            ctx (Any): Context object.
            grad_output (Any): Gradient of the output tensor.

        Returns:
            Any: Gradient of the input tensor.
        """
        
        grad_input = grad_output.clone()
        mean, std = grad_input.mean(), grad_input.std()
        return grad_input.clamp_(mean - 3 * std, mean + 3 * std)


class CliffordQuantizer(object):
    def __init__(self):
        pass

    # straight-through estimator
    @staticmethod
    def quantize_sse(params):
        """
        Apply straight-through estimator quantization to the given parameters.

        Args:
            params (torch.Tensor): Input parameters.

        Returns:
            torch.Tensor: Quantized parameters.
        """
        
        param = params[0][0]
        param = param % (2 * np.pi)
        param = np.pi / 2 * QuantizeFunction.apply(param / (np.pi / 2))
        params = param.unsqueeze(0).unsqueeze(0)
        return params
