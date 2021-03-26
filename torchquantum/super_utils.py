import numpy as np
import torch.nn as nn


class ArchSampler(object):
    def __init__(self, model: nn.Module, strategy=None):
        self.model = model
        self.arch_space = model.arch_space
        # subnetwork sampling strategy
        # None, limit_diff, progressive
        self.strategy = strategy
        self.sample_arch_old = None

    def get_uniform_sample_arch(self):
        sample_arch = []
        if self.strategy['name'] is None or self.sample_arch_old is None:
            for layer_arch_space in self.arch_space:
                layer_arch = np.random.choice(layer_arch_space)
                sample_arch.append(layer_arch)
        elif self.strategy['name'] == 'limit_diff':
            """
            limited differences between architectures of two consecutive 
            samples
            """
            sample_arch = self.sample_arch_old.copy()
            n_diffs = self.strategy['n_diffs']
            assert n_diffs <= len(self.arch_space)
            diff_parts_idx = np.random.choice(np.arange(len(self.arch_space)),
                                              n_diffs,
                                              replace=False)

            for idx in diff_parts_idx:
                sample_arch[idx] = np.random.choice(self.arch_space[idx])
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not "
                                      f"supported.")

        self.sample_arch_old = sample_arch

        return sample_arch

    def get_named_sample_arch(self, name):
        sample_arch = []
        if name == 'smallest':
            for layer_arch_space in self.arch_space:
                layer_arch = layer_arch_space[0]
                sample_arch.append(layer_arch)
        elif name == 'largest':
            for layer_arch_space in self.arch_space:
                layer_arch = layer_arch_space[-1]
                sample_arch.append(layer_arch)
        elif name == 'middle':
            for layer_arch_space in self.arch_space:
                layer_arch = layer_arch_space[len(layer_arch_space) // 2]
                sample_arch.append(layer_arch)
        else:
            raise NotImplementedError(name)
        return sample_arch
