import numpy as np
import torch.nn as nn


class ConfigSampler(object):
    def __init__(self, model: nn.Module):
        self.model = model
        self.config_space = model.config_space

    def get_sample_config(self):
        sample_config = []
        for layer_config_space in self.config_space:
            layer_config = np.random.choice(layer_config_space)
            sample_config.append(layer_config)

        return sample_config
