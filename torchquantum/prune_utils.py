"""
Description: utils for pruning
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-03-24 21:52:50
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-03-24 22:33:25
"""

import torch
import torch.nn.utils.prune
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


__all__ = ["PhaseL1UnstructuredPruningMethod", "ThresholdScheduler"]


class PhaseL1UnstructuredPruningMethod(torch.nn.utils.prune.BasePruningMethod):
    """
    Prune rotation phases which is close to -2pi, 0, +2pi, ..
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        super().__init__()
        # Check range of validity of pruning amount
        # noinspection PyProtectedMember
        torch.nn.utils.prune._validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        t = t % (2 * np.pi)
        t[t > np.pi] -= 2 * np.pi

        tensor_size = t.numel()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        # noinspection PyProtectedMember
        nparams_toprune = torch.nn.utils.prune._compute_nparams_toprune(
            self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        # noinspection PyProtectedMember
        torch.nn.utils.prune._validate_pruning_amount(nparams_toprune,
                                                      tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune,
                              largest=False)
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask


class ThresholdScheduler(object):
    """
    Smooth increasing threshold with tensorflow model pruning scheduler
    """

    def __init__(self, step_beg, step_end, thres_beg, thres_end):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.compat.v1.enable_eager_execution(config=config)
        self.step_beg = step_beg
        self.step_end = step_end
        self.thres_beg = thres_beg
        self.thres_end = thres_end
        if thres_beg < thres_end:
            self.thres_min = thres_beg
            self.thres_range = (thres_end - thres_beg)
            self.descend = False

        else:
            self.thres_min = thres_end
            self.thres_range = (thres_beg - thres_end)
            self.descend = True

        self.pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0, final_sparsity=0.9999999,
            begin_step=self.step_beg, end_step=self.step_end)
        self.global_step = 0

    def step(self):
        if self.global_step < self.step_beg:
            return self.thres_beg
        elif self.global_step > self.step_end:
            return self.thres_end
        res_norm = self.pruning_schedule(self.global_step)[1].numpy()
        if not self.descend:
            res = res_norm * self.thres_range + self.thres_beg
        else:
            res = self.thres_beg - res_norm * self.thres_range

        if np.abs(res - self.thres_end) <= 1e-6:
            res = self.thres_end
        self.global_step += 1
        return res
