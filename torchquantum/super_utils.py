import numpy as np
import torch.nn as nn


class ConfigSampler(object):
    def __init__(self, model: nn.Module, n_comb=None):
        self.model = model
        self.config_space = model.config_space
        self.n_comb = n_comb

    def get_sample_config(self):
        sample_config = {}
        for name, layers in self.config_space.items():
            sample_config[name] = []
            for layer in layers:
                layer_config = {}
                for space_name, spaces in layer.items():
                    layer_config[space_name] = []
                    for space in spaces:
                        # select the number of elements in combinations
                        if self.n_comb is not None:
                            n_comb = self.n_comb
                        else:
                            n_comb = np.random.randint(1, len(space) + 1)
                        comb = np.random.choice(space, n_comb, replace=False)
                        layer_config[space_name].append(comb)
                sample_config[name].append(layer_config)

        return sample_config
