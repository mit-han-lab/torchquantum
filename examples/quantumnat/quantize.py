'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-05-09 21:28:08
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-05-11 01:19:11
'''
from typing import Optional

import torch
from torch import Tensor
from torch.types import Device
from torchpack.utils.logging import logger


__all__ = ["PACTActivationQuantizer"]


# PACT activation: https://arxiv.org/pdf/1805.06085.pdf


class PACTQuantFunc(torch.autograd.Function):
    r"""PACT (PArametrized Clipping acTivation) quantization function for activations.
        Implements a :py:class:`torch.autograd.Function` for quantizing activations in :math:`Q` bits using the PACT strategy.
        In forward propagation, the function is defined as

        .. math::
            \mathbf{y} = f(\mathbf{x}) = 1/\varepsilon \cdot \left\lfloor\mathrm{clip}_{ [0,\alpha) } (\mathbf{x})\right\rfloor \cdot \varepsilon

        where :math:`\varepsilon` is the quantization precision:

        .. math::
            \varepsilon = \alpha / (2^Q - 1)

        In backward propagation, using the Straight-Through Estimator, the gradient of the function is defined as

        .. math::
            \mathbf{\nabla}_\mathbf{x} \mathcal{L} &\doteq \mathbf{\nabla}_\mathbf{y} \mathcal{L}

        It can be applied by using its static `.apply` method:

    :param input: the tensor containing :math:`x`, the activations to be quantized.
    :type  input: `torch.Tensor`
    :param eps: the precomputed value of :math:`\varepsilon`.
    :type  eps: `torch.Tensor` or float
    :param alpha: the value of :math:`\alpha`.
    :type  alpha: `torch.Tensor` or float
    :param delta: constant to sum to `eps` for numerical stability (default unused, 0 ).
    :type  delta: `torch.Tensor` or float

    :return: The quantized input activations tensor.
    :rtype:  `torch.Tensor`
    """

    @staticmethod
    def forward(ctx, input, level, alpha, quant_noise_mask, lower_bound,
                upper_bound):
        # where_input_clipped = (input < -1) | (input > alpha)
        # where_input_ltalpha = (input < alpha)
        # ctx.save_for_backward(where_input_clipped, where_input_ltalpha)
        # upper_thres = alpha.data[0]-eps.data[0]
        input = input.clamp(lower_bound, upper_bound)
        input = input - lower_bound
        eps = (upper_bound - lower_bound) / (level - 1)
        input_q = (input / eps).round() * eps + lower_bound

        # input_q = input.div(eps).floor_().mul_(eps)

        if quant_noise_mask is not None:
            return input_q.data.sub_(input.data).masked_fill_(quant_noise_mask, 0).add_(input)
        else:
            return input_q

    @staticmethod
    def backward(ctx, grad_output):
        # see Hubara et al., Section 2.3
        # where_input_clipped, where_input_ltalpha = ctx.saved_tensors
        # grad_input = grad_output.masked_fill(where_input_clipped, 0)
        # if ctx.needs_input_grad[2]:
        #     grad_alpha = grad_output.masked_fill(
        #         where_input_ltalpha, 0).sum().expand(1)
        # else:
        #     grad_alpha = None
        grad_input = grad_output
        return grad_input, None, None, None, None, None


pact_quantize = PACTQuantFunc.apply


class PACTActivationQuantizer(torch.nn.Module):
    r"""PACT (PArametrized Clipping acTivation) activation.
    Implements a :py:class:`torch.nn.Module` to implement PACT-style activations. It is meant to replace :py:class:`torch.nn.ReLU`, :py:class:`torch.nn.ReLU6` and
    similar activations in a PACT-quantized network.
    This layer can also operate in a special mode, defined by the `statistics_only` member, in which the layer runs in
    forward-prop without quantization, collecting statistics on the activations that can then be
    used to reset the value of :math:`\alpha`.
    In this mode, the layer collects:
    - tensor-wise maximum value ever seen
    - running average with momentum 0.9
    - running variance with momentum 0.9
    """

    def __init__(self, module: torch.nn.Module, precision: Optional[float]=None, level=None, alpha: float = 1.0, backprop_alpha: bool = True,
                 statistics_only: bool = False, leaky: Optional[float] =
                 None, quant_ratio: float = 1.0, device: Device =
                 torch.device("cuda"), lower_bound=-2, upper_bound=2) -> None:
        r"""Constructor. Initializes a :py:class:`torch.nn.Parameter` for :math:`\alpha` and sets
            up the initial value of the `statistics_only` member.
        :param precision: instance defining the current quantization level (default `None`).
        :type  precision: : bitwidth
        :param alpha: the value of :math:`\alpha`.
        :type  alpha: `torch.Tensor` or float
        :param backprop_alpha: default `True`; if `False`, do not update the value of `\alpha` with backpropagation.
        :type  backprop_alpha: bool
        :param statistics_only: initialization value of `statistics_only` member.
        :type  statistics_only: bool
        :param leaky: leaky relu alpha
        :type  leaky: float
        :param quant_ratio: quantization ratio used in QuantNoise [ICLR'21]
        :type  quant_ratio: float
        """

        super().__init__()
        self.module = module
        self.precision = precision
        self.level = level
        self.device = device
        self.alpha = torch.nn.Parameter(torch.tensor(
            (alpha,), device=device), requires_grad=backprop_alpha)
        self.alpha_p = alpha
        self.statistics_only = statistics_only
        self.deployment = False
        self.eps_in = None
        self.leaky = leaky
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # these are only used to gather statistics
        self.max = torch.nn.Parameter(torch.zeros_like(
            self.alpha.data), requires_grad=False)
        self.min = torch.nn.Parameter(torch.zeros_like(
            self.alpha.data), requires_grad=False)
        self.running_mean = torch.nn.Parameter(
            torch.zeros_like(self.alpha.data), requires_grad=False)
        self.running_var = torch.nn.Parameter(
            torch.ones_like(self.alpha.data),  requires_grad=False)

        self.precise = False

        # set quant noise ratio
        self.set_quant_ratio(quant_ratio)

        ## quantization hook
        self.handle = None
        # self.register_hook()

    def set_static_precision(self, limit_at_32_bits: bool = True, **kwargs) -> None:
        r"""Sets static parameters used only for deployment.
        """
        # item() --> conversion to float
        # apparently causes a slight, but not invisibile, numerical divergence
        # between FQ and QD stages
        self.eps_static = self.alpha.clone().detach()/(2.0**(self.precision)-1)
        self.alpha_static = self.alpha.clone().detach()
        # D is selected as a power-of-two
        D = 2.0**torch.ceil(torch.log2(self.requantization_factor *
                                       self.eps_static / self.eps_in))
        if not limit_at_32_bits:
            self.D = D
        else:
            self.D = min(D, 2.0**(32-1-(self.precision)))

    def get_output_eps(self, eps_in: Tensor) -> Tensor:
        r"""Get the output quantum (:math:`\varepsilon`) given the input one.
        :param eps_in: input quantum :math:`\varepsilon_{in}`.
        :type  eps_in: :py:class:`torch.Tensor`
        :return: output quantum :math:`\varepsilon_{out}`.
        :rtype:  :py:class:`torch.Tensor`
        """

        return self.alpha/(2.0**(self.precision)-1)

    def reset_alpha(self, use_max: bool = True, nb_std: float = 5.0) -> None:
        r"""Reset the value of :math:`\alpha`. If `use_max` is `True`, then the highest tensor-wise value collected
            in the statistics collection phase is used. If `False`, the collected standard deviation multiplied by
            `nb_std` is used as a parameter
        :param use_max: if True, use the tensor-wise maximum value collected in the statistics run as new :math:`\alpha` (default True).
        :type  use_max: bool
        :param nb_std: number of standard deviations to be used to initialize :math:`\alpha` if `use_max` is False.
        :type  nb_std: float
        """

        if use_max:
            self.alpha.data.copy_(self.max)
        else:
            self.alpha.data.copy_(self.running_var.data.sqrt().mul(nb_std))

    def get_statistics(self):
        r"""Returns the statistics collected up to now.

        :return: The collected statistics (maximum, running average, running variance).
        :rtype:  tuple of floats
        """
        return self.max.item(), self.running_mean.item(), self.running_var.item()

    def set_quant_ratio(self, quant_ratio=None):
        if(quant_ratio is None):
            # get recommended value
            quant_ratio = [None, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.83,
                           0.86, 0.89, 0.92, 0.95, 0.98, 0.99, 1][min(self.precision, 16)]
        assert 0 <= quant_ratio <= 1, logger.error(
            f"Wrong quant ratio. Must in [0,1], but got {quant_ratio}")
        self.quant_ratio = quant_ratio

    def register_hook(self):

        def quantize_hook(module, x, y):
            r"""Forward-prop function for PACT-quantized activations.

            See :py:class:`nemo.quant.pact_quant.PACT_QuantFunc` for details on the normal operation performed by this layer.
            In statistics mode, it uses a normal ReLU and collects statistics in the background.
            :param x: input activations tensor.
            :type  x: :py:class:`torch.Tensor`

            :return: output activations tensor.
            :rtype:  :py:class:`torch.Tensor`
            """

            if self.statistics_only:
                if self.leaky is None:
                    y = torch.nn.functional.relu(y, inplace=True)
                else:
                    y = torch.nn.functional.leaky_relu(y, self.leaky, inplace=True)
                with torch.no_grad():
                    self.max[:] = max(self.max.item(), y.max())
                    self.min[:] = min(self.min.item(), y.min())
                    self.running_mean[:] = 0.9 * \
                        self.running_mean.item() + 0.1 * y.mean()
                    self.running_var[:] = 0.9 * \
                        self.running_var.item() + 0.1 * y.std()*y.std()
                return y
            else:
                # QuantNoise ICLR 2021
                if(self.quant_ratio < 1 and module.training):
                    # implementation from fairseq
                    # must fully quantize during inference
                    quant_noise_mask = torch.empty_like(
                        y, dtype=torch.bool).bernoulli_(1-self.quant_ratio)
                else:
                    quant_noise_mask = None
                if self.level is not None:
                    level = self.level
                else:
                    level = 2 ** self.precision
                # eps = self.alpha/(2.0**(self.precision)-1)
                return pact_quantize(y, level, self.alpha, quant_noise_mask,
                                     self.lower_bound, self.upper_bound)

        # register hook
        self.handle = self.module.register_forward_hook(quantize_hook)
        return self.handle

    def remove_hook(self) -> None:
        ## remove the forward hook
        if(self.handle is not None):
            self.handle.remove()


if __name__ == "__main__":
    import pdb
    pdb.set_trace()
    device = torch.device("cuda")
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x):
            y = x + 0.3
            return y
    model = Model().to(device)
    model.train()
    quantizer = PACTActivationQuantizer(module=model, precision=4,
                                        quant_ratio=0.1, device=device,
                                        backprop_alpha=False)
    quantizer.set_quant_ratio(0.8)
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    x = torch.randn(4,4, device=device, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print(x)
    print(y)
    print(quantizer.alpha.data)
    print(quantizer.alpha.grad)

    