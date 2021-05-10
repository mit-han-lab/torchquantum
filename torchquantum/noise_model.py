import numpy as np
import torch.nn as nn
from typing import Iterable
from torchpack.utils.logging import logger
from qiskit.providers.aer.noise import NoiseModel

__all__ = ['NoiseModelTQ']


class NoiseModelTQ(object):
    def __init__(self, backend):
        self.noise_model = NoiseModel.from_backend(backend)
        self.noise_model_dict = self.noise_model.to_dict()

    def sample_noise_op(self, in_op):
        pass
