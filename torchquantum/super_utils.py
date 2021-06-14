import numpy as np
import torch.nn as nn
from typing import Iterable
from torchpack.utils.logging import logger
from torchpack.utils.config import configs


class ArchSampler(object):
    def __init__(self, model: nn.Module, strategy=None,
                 n_layers_per_block=None):
        self.model = model
        self.total_steps = None
        self.arch_space = model.arch_space
        # subnetwork sampling strategy
        # None, limit_diff, progressive
        self.strategy = strategy
        self.sample_arch_old = None
        self.arch_space_stage = None
        self.n_ops_smallest = 0
        self.n_ops_largest = 0

        self.is_block_based = False
        self.n_layers_per_block = n_layers_per_block
        if n_layers_per_block is not None:
            self.is_block_based = True

        self.get_space_stats()
        self.step = 0

        self.n_ops_per_chunk = None
        if strategy['name'] == 'progressive':
            self.get_n_ops_per_chunk()

        self.sample_n_ops = None
        self.current_stage = 0
        self.current_chunk = 0

    def set_total_steps(self, total_steps):
        self.total_steps = total_steps

    def get_n_ops_per_chunk(self):
        """separate the space to several subspace"""
        n_chunks = self.strategy['n_chunks']
        if self.strategy['chunk_mode'] == 'same_interval':
            logger.warning(f"same_interval chunking may cause extra long "
                           f"time to sample a sub network because of the "
                           f"Central Limit Theorem of n_ops in a subnet")
            self.n_ops_per_chunk = list(np.linspace(self.n_ops_smallest,
                                                    self.n_ops_largest,
                                                    n_chunks + 1).astype(int))
        elif self.strategy['chunk_mode'] == 'same_n_samples':
            logger.info("estimating the chunks...")
            n_ops_all = []
            n_chunk_est_samples = self.strategy['n_chunk_est_samples']
            for k in range(n_chunk_est_samples):
                sample_arch = self.get_random_sample_arch()
                n_ops_all.append(self.get_sample_stats(sample_arch))
            n_ops_all.sort()
            idx_all = np.linspace(0, n_chunk_est_samples - 1, n_chunks +
                                  1).astype(int)
            self.n_ops_per_chunk = [n_ops_all[idx] for idx in idx_all]
            self.n_ops_per_chunk[0] = self.n_ops_smallest
            self.n_ops_per_chunk[-1] = self.n_ops_largest
        else:
            raise NotImplementedError(
                f"chunk mode {self.strategy['chunk_mode']} not supported.")

    def get_sample_stats(self, sample_arch):
        n_ops = 0
        if self.is_block_based:
            layers_arch = sample_arch[:-1]
            block_arch = sample_arch[-1]
        else:
            layers_arch = self.arch_space
            block_arch = None

        for k, layer_arch in enumerate(layers_arch):
            if not isinstance(layer_arch, Iterable):
                # share front layer
                if not self.is_block_based or k < block_arch * \
                        self.n_layers_per_block:
                    n_ops += layer_arch
            else:
                # arbitrary layer
                if not self.is_block_based or k < block_arch * \
                        self.n_layers_per_block:
                    n_ops += len(layer_arch)

        return n_ops

    def get_space_stats(self):
        """get the max number and smallest number of ops in the space"""
        if self.is_block_based:
            layers_space = self.arch_space[:-1]
            block_space = self.arch_space[-1]
        else:
            layers_space = self.arch_space
            block_space = None

        for k, layer_space in enumerate(layers_space):
            if not isinstance(layer_space[0], Iterable):
                # share front layer
                self.n_ops_largest += max(layer_space)
                if not self.is_block_based or k < min(block_space) * \
                        self.n_layers_per_block:
                    self.n_ops_smallest += min(layer_space)
            else:
                # arbitrary layer
                self.n_ops_largest += max(list(map(len, layer_space)))
                if not self.is_block_based or k < min(block_space) * \
                        self.n_layers_per_block:
                    self.n_ops_smallest += min(list(map(len, layer_space)))

    def get_random_sample_arch(self):
        sample_arch = []
        for layer_arch_space in self.arch_space:
            layer_arch = np.random.choice(layer_arch_space)
            sample_arch.append(layer_arch)
        return sample_arch

    def get_uniform_sample_arch(self):
        if self.strategy['name'] == 'plain' or (self.strategy['name'] ==
                                                'limit_diff' and
                                                self.sample_arch_old is None):
            sample_arch = self.get_random_sample_arch()
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
        elif self.strategy['name'] == 'progressive':
            """
            different stages have different model capacity.
            In different stage the total number of gates is specified
            """
            n_stages = self.strategy['n_stages']
            n_chunks = self.strategy['n_chunks']
            while True:
                sample_arch = self.get_random_sample_arch()
                n_ops = self.get_sample_stats(sample_arch)
                current_stage = int(self.step // (self.total_steps / n_stages))
                current_chunk = current_stage % n_chunks
                self.current_chunk = current_chunk
                self.current_stage = current_stage
                if self.strategy['subspace_mode'] == 'expand':
                    # the subspace size is expanding
                    if self.strategy['direction'] == 'top_down':
                        if n_ops >= list(reversed(self.n_ops_per_chunk))[
                                current_chunk + 1]:
                            break
                    elif self.strategy['direction'] == 'bottom_up':
                        if n_ops <= self.n_ops_per_chunk[current_chunk + 1]:
                            break
                    else:
                        raise NotImplementedError(
                            f"Direction mode {self.strategy['direction']} "
                            f"not supported.")
                elif self.strategy['subspace_mode'] == 'same':
                    # the subspace size is the same
                    if self.strategy['direction'] == 'top_down':
                        left = list(reversed(self.n_ops_per_chunk))[
                            current_chunk + 1]
                        right = list(reversed(self.n_ops_per_chunk))[
                            current_chunk]
                    elif self.strategy['direction'] == 'bottom_up':
                        left = self.n_ops_per_chunk[current_chunk]
                        right = self.n_ops_per_chunk[current_chunk + 1]
                    else:
                        raise NotImplementedError(
                            f"Direction mode {self.strategy['direction']} "
                            f"not supported.")

                    if left <= n_ops <= right:
                        break
                else:
                    raise NotImplementedError(
                        f"Subspace mode {self.strategy['subspace_mode']} "
                        f"not supported.")
        elif self.strategy['name'] == 'limit_diff_expanding':
            """
            shrink the overall number of blocks and in-block size
            """
            if self.sample_arch_old is None:
                self.sample_arch_old = get_named_sample_arch(
                    self.arch_space, 'largest')
            sample_arch = self.sample_arch_old.copy()
            n_stages = self.strategy['n_stages']
            n_chunks = self.strategy['n_chunks']
            n_diffs = self.strategy['n_diffs']
            assert n_diffs <= len(self.arch_space)

            current_stage = int(self.step // (self.total_steps / n_stages))
            current_chunk = current_stage % n_chunks
            self.current_stage = current_stage
            self.current_chunk = current_chunk
            diff_parts_idx = np.random.choice(np.arange(len(self.arch_space)),
                                              n_diffs,
                                              replace=False)

            for idx in diff_parts_idx:
                layer_arch_space = self.arch_space[idx]
                n_choices = len(layer_arch_space)
                new_space = layer_arch_space[
                    int(round((n_choices - 1) * (1 - (current_chunk + 1) / (
                        n_chunks)))):]
                if len(new_space) == 1:
                    sample_arch[idx] = new_space[0]
                else:
                    sample_arch[idx] = np.random.choice(new_space)
        elif self.strategy['name'] == 'ldiff_blkexpand':
            """
            shrink the overall number of blocks only
            """
            if self.sample_arch_old is None:
                self.sample_arch_old = get_named_sample_arch(
                    self.arch_space, 'largest')
            sample_arch = self.sample_arch_old.copy()
            n_stages = self.strategy['n_stages']
            n_chunks = self.strategy['n_chunks']
            n_diffs = self.strategy['n_diffs']
            assert n_diffs <= len(self.arch_space)

            current_stage = int(self.step // (self.total_steps / n_stages))
            if current_stage >= n_chunks:
                """
                major difference here, after expanding the space, 
                will not go back
                """
                current_chunk = n_chunks - 1
            else:
                current_chunk = current_stage
            self.current_stage = current_stage
            self.current_chunk = current_chunk
            diff_parts_idx = np.random.choice(np.arange(len(self.arch_space)),
                                              n_diffs,
                                              replace=False)
            new_arch_space = self.arch_space.copy()
            n_blk_choices = len(new_arch_space[-1])
            new_arch_space[-1] = new_arch_space[-1][
                int(round((n_blk_choices - 1) * (1 - (current_chunk + 1) / (
                    n_chunks)))):]
            for idx in diff_parts_idx:
                sample_arch[idx] = np.random.choice(new_arch_space[idx])
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not "
                                      f"supported.")

        self.sample_n_ops = self.get_sample_stats(sample_arch)
        self.sample_arch_old = sample_arch
        self.step += 1

        return sample_arch


def get_named_sample_arch(arch_space, name):
    """
    examples:
    - blk1_ratio0.5 means 1 block and each layers' arch is the 0.5 of all
    choices
    - blk8_ratio0.3 means 8 block and each layers' arch is 0.3 of all choices
    - ratio0.4 means 0.4 for each arch choice, including blk if exists
    """

    sample_arch = []
    if name == 'smallest':
        for layer_arch_space in arch_space:
            layer_arch = layer_arch_space[0]
            sample_arch.append(layer_arch)
    elif name == 'largest':
        for layer_arch_space in arch_space:
            layer_arch = layer_arch_space[-1]
            sample_arch.append(layer_arch)
    elif name == 'middle':
        for layer_arch_space in arch_space:
            layer_arch = layer_arch_space[len(layer_arch_space) // 2]
            sample_arch.append(layer_arch)
    elif name.startswith('blk'):
        # decode the block and ratio
        n_block = eval(name.split('_')[0].replace('blk', ''))
        ratio = eval(name.split('_')[1].replace('ratio', ''))
        assert ratio <= 1
        for layer_arch_space in arch_space[:-1]:
            layer_arch = layer_arch_space[
                int(round((len(layer_arch_space) - 1) * ratio))]
            sample_arch.append(layer_arch)
        sample_arch.append(n_block)
    elif name.startswith('ratio'):
        # decode the numerator and denominator
        ratio = eval(name.replace('ratio', ''))
        assert ratio <= 1
        for layer_arch_space in arch_space:
            layer_arch = layer_arch_space[
                int(round((len(layer_arch_space) - 1) * ratio))]
            sample_arch.append(layer_arch)
    elif name.startswith('super4digit_arbitrary_fc1'):
        """specific sampled arch for super4digit_arbitrary_fc1 design space"""
        arch_dict = {
            'blk1_rand0': [[2], [[1, 2], [2, 3], [3, 0]], [1, 2, 3], [[2, 3]], [0, 1, 2, 3], [[1, 2]], [3], [[0, 1], [1, 2]], [0, 2], [[0, 1], [1, 2], [2, 3], [3, 0]], [0, 1, 3], [[1, 2], [2, 3]], [0, 1, 3], [[0, 1], [1, 2], [2, 3]], [1, 2, 3], [[0, 1], [1, 2], [2, 3]], 1],
            'blk1_rand1': [[0, 2], [[1, 2]], [0, 2, 3], [[2, 3]], [0, 2], [[0, 1], [1, 2], [3, 0]], [0, 1], [[0, 1], [3, 0]], [1, 3], [[1, 2], [3, 0]], [1], [[0, 1], [1, 2], [2, 3]], [0, 1], [[0, 1], [1, 2], [3, 0]], [1], [[2, 3]], 1],
            'blk1_rand2': [[0, 1, 3], [[0, 1], [2, 3], [3, 0]], [1, 3], [[2, 3], [3, 0]], [1, 2, 3], [[1, 2]], [1, 3], [[0, 1], [1, 2], [2, 3]], [0], [[2, 3], [3, 0]], [3], [[0, 1], [1, 2], [2, 3], [3, 0]], [0, 2], [[2, 3], [3, 0]], [0, 2], [[0, 1]], 1],
            'blk1_rand3': [[0, 1, 2, 3], [[0, 1], [2, 3]], [0, 2, 3], [[0, 1], [3, 0]], [3], [[0, 1], [2, 3]], [0, 1, 2, 3], [[2, 3]], [2], [[0, 1], [3, 0]], [1], [[1, 2], [3, 0]], [1, 3], [[0, 1], [1, 2]], [0, 2, 3], [[1, 2], [2, 3]], 1],
            'blk2_rand0': [[3], [[2, 3], [3, 0]], [0, 3], [[1, 2], [2, 3]], [1], [[0, 1], [1, 2], [2, 3]], [2, 3], [[1, 2]], [2, 3], [[0, 1], [2, 3]], [0, 3], [[2, 3]], [2, 3], [[0, 1], [2, 3]], [3], [[0, 1], [2, 3]], 2],
            'blk2_rand1': [[2, 3], [[2, 3]], [0, 1], [[0, 1]], [0, 3], [[0, 1], [1, 2], [2, 3], [3, 0]], [3], [[0, 1], [2, 3], [3, 0]], [0, 1, 3], [[2, 3]], [0, 1], [[0, 1]], [0, 2], [[1, 2]], [1, 2, 3], [[0, 1], [1, 2]], 2],
            'blk2_rand2': [[0, 1, 2], [[1, 2], [2, 3], [3, 0]], [2], [[0, 1], [1, 2], [2, 3], [3, 0]], [0, 2, 3], [[0, 1], [1, 2], [2, 3], [3, 0]], [0, 1, 2], [[0, 1], [2, 3], [3, 0]], [2, 3], [[0, 1], [3, 0]], [0, 1], [[0, 1], [1, 2], [2, 3]], [0, 2], [[0, 1]], [1, 2], [[0, 1], [1, 2], [3, 0]], 2],
            'blk2_rand3': [[0, 1, 2, 3], [[0, 1], [1, 2], [2, 3]], [2], [[1, 2], [2, 3]], [1, 2], [[0, 1], [1, 2], [2, 3], [3, 0]], [0, 1, 3], [[0, 1], [2, 3], [3, 0]], [3], [[1, 2], [2, 3]], [0, 1, 3], [[2, 3]], [1, 2, 3], [[0, 1], [1, 2], [2, 3]], [0, 1, 2], [[2, 3]], 2],
            'blk3_rand0': [[2], [[2, 3]], [0, 2, 3], [[1, 2], [2, 3], [3, 0]], [1], [[0, 1], [2, 3], [3, 0]], [2], [[0, 1], [1, 2], [3, 0]], [1, 2, 3], [[2, 3]], [2, 3], [[1, 2], [3, 0]], [1], [[1, 2], [3, 0]], [2, 3], [[0, 1], [1, 2]], 3],
            'blk3_rand1': [[1, 2], [[2, 3]], [0, 1, 2], [[0, 1], [2, 3]], [0], [[0, 1], [2, 3]], [0], [[1, 2], [2, 3]], [1, 3], [[0, 1], [1, 2], [2, 3]], [2], [[0, 1]], [1, 2], [[3, 0]], [1, 2, 3], [[0, 1], [2, 3]], 3],
            'blk3_rand2': [[0, 2, 3], [[2, 3], [3, 0]], [0, 1, 2, 3], [[1, 2]], [0, 1, 3], [[2, 3]], [2, 3], [[1, 2], [2, 3], [3, 0]], [0], [[1, 2]], [1, 2], [[0, 1], [2, 3], [3, 0]], [1, 3], [[0, 1], [2, 3], [3, 0]], [2, 3], [[1, 2], [2, 3], [3, 0]], 3],
            'blk3_rand3': [[0, 1, 2, 3], [[2, 3]], [0, 1, 2], [[0, 1], [2, 3]], [2], [[3, 0]], [1, 3], [[0, 1], [1, 2], [3, 0]], [0, 2], [[1, 2], [3, 0]], [0, 1], [[1, 2]], [2], [[0, 1], [1, 2], [3, 0]], [0, 1], [[0, 1], [2, 3], [3, 0]], 3],
            'blk4_rand0': [[0], [[2, 3], [3, 0]], [2, 3], [[2, 3], [3, 0]], [1, 3], [[1, 2], [2, 3]], [2, 3], [[2, 3]], [1], [[2, 3]], [0, 1, 2, 3], [[1, 2]], [2], [[1, 2], [2, 3], [3, 0]], [1, 2, 3], [[0, 1], [1, 2], [3, 0]], 4],
            'blk4_rand1': [[0, 2], [[2, 3], [3, 0]], [0, 3], [[0, 1]], [1], [[0, 1]], [0, 2], [[1, 2], [2, 3], [3, 0]], [1, 2, 3], [[0, 1], [1, 2], [3, 0]], [1, 2], [[0, 1], [1, 2], [2, 3]], [0, 1, 3], [[0, 1], [2, 3]], [0, 1, 3], [[0, 1]], 4],
            'blk4_rand2': [[0, 1, 2], [[0, 1], [1, 2]], [2, 3], [[2, 3]], [1, 3], [[1, 2], [2, 3], [3, 0]], [1, 2, 3], [[1, 2], [3, 0]], [0, 1, 2], [[0, 1], [3, 0]], [0, 1, 2], [[2, 3]], [0], [[3, 0]], [0, 1], [[1, 2], [3, 0]], 4],
            'blk4_rand3': [[0, 1, 2, 3], [[0, 1], [1, 2]], [0, 1, 3], [[0, 1]], [0, 1, 2], [[0, 1], [1, 2], [3, 0]], [1, 2, 3], [[2, 3], [3, 0]], [0, 1, 2, 3], [[2, 3], [3, 0]], [1, 2, 3], [[3, 0]], [1, 3], [[0, 1], [1, 2], [2, 3]], [0, 1, 2, 3], [[0, 1]], 4],
            'blk5_rand0': [[3], [[0, 1], [1, 2], [2, 3]], [0, 1, 3], [[0, 1], [3, 0]], [1, 2], [[0, 1], [1, 2], [2, 3]], [0, 1], [[0, 1], [1, 2], [2, 3]], [1, 2, 3], [[2, 3], [3, 0]], [0, 1, 2], [[1, 2], [2, 3], [3, 0]], [2], [[0, 1], [2, 3], [3, 0]], [3], [[0, 1], [1, 2], [2, 3]], 5],
            'blk5_rand1': [[0, 1], [[0, 1], [1, 2], [2, 3]], [0, 1, 2], [[1, 2]], [0, 2, 3], [[1, 2], [2, 3]], [0, 1, 3], [[0, 1], [3, 0]], [0, 2], [[0, 1], [1, 2], [3, 0]], [0, 3], [[1, 2], [3, 0]], [1], [[0, 1], [1, 2], [2, 3], [3, 0]], [1], [[1, 2], [2, 3], [3, 0]], 5],
            'blk5_rand2': [[0, 2, 3], [[0, 1], [1, 2]], [0, 1], [[1, 2], [3, 0]], [2], [[1, 2], [3, 0]], [0, 2], [[0, 1], [2, 3], [3, 0]], [1, 2, 3], [[0, 1], [3, 0]], [2, 3], [[2, 3], [3, 0]], [0, 3], [[1, 2], [2, 3]], [1], [[1, 2], [2, 3]], 5],
            'blk5_rand3': [[0, 1, 2, 3], [[1, 2], [2, 3], [3, 0]], [0, 1, 3], [[1, 2], [3, 0]], [0, 1], [[2, 3]], [0, 1], [[1, 2], [3, 0]], [2], [[1, 2]], [2, 3], [[0, 1], [1, 2], [2, 3]], [0, 1, 2], [[0, 1], [1, 2]], [0, 3], [[0, 1], [1, 2], [2, 3], [3, 0]], 5],
            'blk6_rand0': [[2], [[1, 2]], [0, 2, 3], [[1, 2], [3, 0]], [0, 1, 2], [[1, 2], [2, 3]], [0], [[0, 1], [1, 2], [3, 0]], [0, 3], [[1, 2], [2, 3], [3, 0]], [0, 2, 3], [[1, 2], [2, 3], [3, 0]], [0, 2, 3], [[0, 1], [1, 2], [3, 0]], [0, 1, 3], [[0, 1]], 6],
            'blk6_rand1': [[1, 3], [[2, 3]], [2, 3], [[1, 2]], [2], [[1, 2], [2, 3], [3, 0]], [0, 2, 3], [[0, 1]], [0, 2, 3], [[0, 1], [1, 2], [2, 3]], [2], [[1, 2], [2, 3], [3, 0]], [0, 2], [[2, 3]], [0, 3], [[1, 2], [2, 3], [3, 0]], 6],
            'blk6_rand2': [[0, 1, 3], [[1, 2], [2, 3], [3, 0]], [0, 3], [[3, 0]], [0, 1, 3], [[0, 1], [1, 2], [3, 0]], [1], [[0, 1]], [0, 1, 3], [[0, 1], [1, 2]], [2], [[0, 1], [2, 3]], [1, 2], [[0, 1], [1, 2], [3, 0]], [0], [[1, 2], [2, 3], [3, 0]], 6],
            'blk6_rand3': [[0, 1, 2, 3], [[3, 0]], [1], [[3, 0]], [0, 2], [[0, 1]], [2, 3], [[0, 1]], [3], [[1, 2], [2, 3], [3, 0]], [1], [[0, 1], [1, 2]], [0, 1, 2], [[1, 2], [3, 0]], [0, 2, 3], [[1, 2], [3, 0]], 6],
            'blk7_rand0': [[3], [[1, 2], [2, 3], [3, 0]], [0, 1, 2], [[1, 2], [2, 3]], [1, 3], [[1, 2], [2, 3]], [1], [[1, 2], [3, 0]], [0, 1], [[2, 3]], [0, 3], [[0, 1], [3, 0]], [2, 3], [[1, 2], [2, 3]], [0, 1, 3], [[0, 1], [1, 2], [3, 0]], 7],
            'blk7_rand1': [[0, 3], [[2, 3]], [0, 1, 2, 3], [[1, 2], [3, 0]], [0, 1, 3], [[0, 1]], [3], [[0, 1], [1, 2], [2, 3], [3, 0]], [1, 2], [[1, 2]], [0, 1, 2], [[0, 1], [1, 2]], [0, 1, 3], [[0, 1], [1, 2], [3, 0]], [3], [[1, 2]], 7],
            'blk7_rand2': [[0, 1, 2], [[0, 1]], [2], [[1, 2]], [3], [[0, 1], [1, 2], [2, 3], [3, 0]], [1], [[0, 1], [3, 0]], [0], [[0, 1], [1, 2], [2, 3]], [1], [[1, 2], [2, 3], [3, 0]], [0, 1, 3], [[0, 1], [1, 2], [2, 3]], [1, 2, 3], [[1, 2], [3, 0]], 7],
            'blk7_rand3': [[0, 1, 2, 3], [[2, 3], [3, 0]], [3], [[3, 0]], [0, 1, 3], [[2, 3]], [0, 2, 3], [[1, 2], [2, 3]], [0, 3], [[0, 1], [2, 3]], [1], [[0, 1], [1, 2], [2, 3]], [1, 2, 3], [[0, 1], [1, 2], [2, 3], [3, 0]], [1, 2, 3], [[0, 1], [1, 2]], 7],
            'blk8_rand0': [[0], [[2, 3]], [0], [[0, 1], [3, 0]], [2, 3], [[1, 2], [2, 3]], [0, 3], [[1, 2], [2, 3]], [0, 1, 2], [[0, 1], [1, 2]], [0], [[0, 1], [2, 3]], [0, 1], [[2, 3]], [0, 1], [[2, 3], [3, 0]], 8],
            'blk8_rand1': [[0, 1], [[3, 0]], [1], [[1, 2], [2, 3], [3, 0]], [0, 1, 3], [[3, 0]], [0], [[0, 1], [1, 2]], [3], [[0, 1], [1, 2], [2, 3]], [1, 3], [[1, 2], [2, 3]], [1, 2], [[0, 1], [3, 0]], [2, 3], [[3, 0]], 8],
            'blk8_rand2': [[1, 2, 3], [[1, 2], [2, 3], [3, 0]], [0, 1], [[0, 1], [1, 2], [3, 0]], [0, 1, 2], [[0, 1], [1, 2], [2, 3], [3, 0]], [2], [[0, 1], [2, 3], [3, 0]], [0, 1, 3], [[0, 1], [2, 3], [3, 0]], [1], [[1, 2], [2, 3], [3, 0]], [0, 2], [[0, 1], [2, 3], [3, 0]], [0, 1, 2, 3], [[0, 1], [1, 2], [2, 3]], 8],
            'blk8_rand3': [[0, 1, 2, 3], [[1, 2], [3, 0]], [0, 1, 2, 3], [[0, 1], [3, 0]], [0, 1, 2], [[0, 1], [2, 3], [3, 0]], [1, 3], [[1, 2]], [1, 3], [[0, 1], [1, 2], [2, 3]], [3], [[0, 1]], [3], [[0, 1], [1, 2]], [0, 1, 3], [[2, 3]], 8],
        }
        sample_arch = arch_dict[name.replace('super4digit_arbitrary_fc1_', '')]
    elif name.startswith('sharefront0_'):
        """specific sampled arch for sharefront0_.* design space"""
        if configs.model.arch.q_layer_name in ['u3cu3_s0'] and configs.model.arch.n_wires == 20:
            arch_dict = {
                'blk1_rand0': [7, 20, 1],
                'blk1_rand1': [15, 11, 1],
                'blk1_rand2': [8, 7, 1],
                'blk1_rand3': [19, 11, 1],
                'blk1_rand4': [11, 4, 1],
                'blk1_rand5': [8, 3, 1],
                'blk1_rand6': [2, 12, 1],
                'blk1_rand7': [6, 2, 1],
                'blk1_rand8': [1, 12, 1],
                'blk1_rand9': [12, 17, 1],
                'blk1_rand10': [10, 16, 1],
                'blk1_rand11': [15, 15, 1],
                'blk1_rand12': [19, 12, 1],
                'blk1_rand13': [20, 3, 1],
                'blk1_rand14': [5, 19, 1],
                'blk1_rand15': [7, 9, 1],
            }
        elif configs.model.arch.q_layer_name in ['u3cu3_s0', 'seth_s0', 'farhi_s0']:
            arch_dict = {
                'blk1_rand0': [1, 2, 3, 1, 4, 1, 4, 4, 4, 3, 3, 3, 1, 4, 3, 3, 1],
                'blk1_rand1': [2, 3, 4, 3, 4, 4, 1, 3, 1, 3, 3, 1, 1, 3, 2, 4, 1],
                'blk1_rand2': [3, 2, 2, 3, 3, 1, 3, 3, 2, 2, 4, 1, 3, 3, 4, 3, 1],
                'blk1_rand3': [4, 3, 2, 1, 1, 2, 4, 2, 4, 2, 3, 4, 3, 1, 2, 1, 1],
                'blk1_rand4': [2, 3, 2, 1, 1, 2, 4, 2, 4, 2, 3, 4, 3, 1, 2, 1, 1],
                'blk1_rand5': [4, 3, 1, 3, 1, 1, 1, 2, 4, 1, 1, 2, 1, 3, 3, 4, 1],
                'blk2_rand0': [1, 2, 2, 1, 4, 1, 1, 3, 2, 4, 3, 1, 2, 3, 2, 3, 2],
                'blk2_rand1': [2, 1, 3, 3, 4, 1, 4, 3, 1, 1, 4, 3, 1, 4, 3, 4, 2],
                'blk2_rand2': [3, 4, 3, 2, 3, 3, 3, 4, 4, 3, 4, 1, 1, 4, 4, 3, 2],
                'blk2_rand3': [4, 3, 1, 3, 1, 1, 1, 2, 4, 1, 1, 2, 1, 3, 3, 4, 2],
                'blk2_rand4': [3, 4, 3, 1, 3, 4, 2, 4, 4, 4, 2, 4, 4, 4, 1, 4, 2],
                'blk2_rand5': [3, 1, 4, 1, 1, 3, 4, 4, 1, 2, 3, 3, 3, 3, 4, 4, 2],
                'blk3_rand0': [1, 4, 4, 1, 2, 1, 4, 2, 4, 1, 1, 4, 3, 3, 1, 1, 3],
                'blk3_rand1': [2, 1, 4, 4, 4, 1, 2, 2, 1, 2, 1, 2, 3, 3, 4, 3, 3],
                'blk3_rand2': [3, 1, 4, 4, 1, 1, 1, 1, 4, 2, 1, 1, 4, 4, 3, 1, 3],
                'blk3_rand3': [4, 3, 3, 3, 4, 2, 2, 2, 4, 3, 2, 4, 2, 1, 2, 2, 3],
                'blk3_rand4': [2, 3, 1, 1, 2, 4, 4, 2, 2, 1, 4, 3, 1, 2, 4, 3, 3],
                'blk3_rand5': [3, 3, 1, 2, 1, 1, 2, 2, 4, 2, 3, 1, 1, 3, 2, 3, 3],
                'blk4_rand0': [1, 4, 1, 4, 2, 3, 1, 4, 1, 4, 1, 4, 4, 2, 4, 2, 4],
                'blk4_rand1': [2, 2, 2, 1, 4, 2, 2, 3, 2, 3, 4, 1, 1, 4, 1, 4, 4],
                'blk4_rand2': [3, 3, 2, 4, 4, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 4],
                'blk4_rand3': [4, 2, 3, 3, 2, 2, 3, 3, 2, 3, 1, 1, 2, 4, 2, 2, 4],
                'blk4_rand4': [1, 4, 2, 3, 3, 4, 1, 2, 1, 1, 1, 1, 3, 3, 1, 2, 4],
                'blk4_rand5': [3, 3, 1, 4, 3, 2, 2, 4, 2, 2, 4, 3, 3, 2, 2, 3, 4],
                'blk5_rand0': [1, 4, 1, 2, 3, 1, 4, 2, 1, 4, 4, 4, 1, 1, 1, 3, 5],
                'blk5_rand1': [2, 1, 4, 2, 2, 1, 1, 3, 4, 4, 3, 4, 4, 4, 4, 4, 5],
                'blk5_rand2': [3, 1, 4, 1, 1, 2, 2, 1, 3, 2, 4, 2, 2, 4, 1, 4, 5],
                'blk5_rand3': [4, 1, 3, 1, 2, 2, 4, 2, 3, 1, 4, 1, 1, 3, 2, 2, 5],
                'blk5_rand4': [3, 1, 4, 2, 2, 1, 1, 3, 4, 4, 3, 4, 4, 4, 4, 4, 5],
                'blk5_rand5': [4, 1, 3, 4, 2, 1, 4, 1, 1, 1, 3, 3, 2, 2, 4, 2, 5],
                'blk6_rand0': [1, 4, 3, 3, 4, 1, 2, 3, 1, 1, 2, 3, 3, 1, 1, 2, 6],
                'blk6_rand1': [2, 3, 2, 2, 3, 2, 3, 2, 3, 1, 4, 3, 3, 4, 3, 3, 6],
                'blk6_rand2': [3, 2, 4, 2, 2, 1, 2, 1, 1, 2, 4, 4, 4, 4, 4, 2, 6],
                'blk6_rand3': [4, 2, 4, 3, 4, 3, 2, 2, 2, 2, 2, 3, 3, 2, 4, 1, 6],
                'blk6_rand4': [4, 1, 4, 3, 2, 1, 4, 3, 2, 1, 2, 3, 1, 2, 4, 1, 6],
                'blk6_rand5': [3, 3, 2, 2, 3, 2, 3, 2, 3, 1, 4, 3, 3, 4, 3, 3, 6],
                'blk7_rand0': [1, 3, 3, 3, 3, 4, 1, 4, 4, 3, 2, 2, 4, 4, 3, 4, 7],
                'blk7_rand1': [2, 1, 1, 1, 4, 2, 2, 2, 1, 2, 2, 1, 3, 1, 2, 1, 7],
                'blk7_rand2': [3, 1, 3, 1, 2, 3, 1, 2, 3, 3, 1, 1, 1, 2, 1, 3, 7],
                'blk7_rand3': [4, 2, 2, 3, 1, 4, 3, 3, 4, 2, 3, 2, 1, 2, 1, 2, 7],
                'blk7_rand4': [1, 1, 2, 1, 2, 2, 3, 4, 4, 2, 3, 2, 2, 2, 3, 4, 7],
                'blk7_rand5': [4, 3, 3, 3, 3, 4, 1, 4, 4, 3, 2, 2, 4, 4, 3, 4, 7],
                'blk8_rand0': [1, 1, 4, 1, 4, 1, 2, 1, 4, 4, 4, 4, 4, 3, 3, 1, 8],
                'blk8_rand1': [2, 3, 2, 1, 4, 2, 1, 2, 3, 2, 2, 1, 1, 1, 2, 3, 8],
                'blk8_rand2': [3, 2, 1, 2, 1, 4, 3, 4, 1, 1, 4, 1, 4, 3, 4, 3, 8],
                'blk8_rand3': [4, 4, 1, 3, 1, 3, 1, 3, 1, 4, 2, 1, 3, 4, 1, 1, 8],
                'blk8_rand4': [2, 2, 2, 1, 3, 2, 1, 2, 2, 1, 3, 3, 4, 1, 4, 4, 8],
                'blk8_rand5': [1, 3, 1, 4, 1, 1, 4, 2, 2, 4, 3, 1, 4, 1, 3, 1, 8],
            }
        elif configs.model.arch.q_layer_name in ['barren_s0']:
            arch_dict = {
                'blk1_rand0': [1, 4, 2, 4, 1, 1, 4, 3, 3, 1, 1, 3, 3, 4, 4, 3, 2, 4, 1, 4, 3, 1, 4, 4, 2, 4, 4, 2, 3, 4, 2, 1, 3, 1],
                'blk1_rand1': [2, 4, 3, 3, 2, 1, 2, 1, 3, 4, 1, 1, 1, 4, 3, 4, 3, 4, 1, 4, 1, 4, 4, 2, 1, 2, 1, 2, 3, 4, 1, 1, 1, 1],
                'blk1_rand2': [1, 3, 2, 4, 2, 2, 4, 1, 1, 1, 1, 4, 1, 3, 4, 3, 1, 2, 1, 3, 3, 4, 2, 4, 3, 1, 2, 1, 1, 2, 4, 1, 3, 1],
                'blk1_rand3': [1, 3, 2, 3, 3, 1, 2, 4, 1, 2, 2, 2, 2, 2, 3, 1, 3, 4, 2, 1, 2, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 4, 3, 1],
                'blk1_rand4': [3, 4, 4, 2, 1, 3, 4, 4, 3, 2, 4, 1, 1, 3, 1, 3, 1, 3, 1, 2, 1, 1, 2, 4, 1, 2, 1, 3, 3, 1, 4, 4, 1, 1],
                'blk1_rand5': [3, 3, 1, 3, 3, 2, 3, 4, 1, 2, 2, 4, 1, 1, 2, 3, 2, 1, 4, 2, 1, 2, 3, 2, 2, 1, 1, 1, 2, 3, 4, 4, 1, 1],
                'blk2_rand0': [4, 1, 4, 1, 2, 3, 1, 4, 2, 1, 4, 4, 1, 1, 1, 3, 1, 1, 1, 3, 1, 4, 1, 4, 3, 3, 3, 1, 3, 3, 1, 3, 1, 2],
                'blk2_rand1': [3, 2, 3, 3, 3, 4, 4, 3, 1, 1, 4, 4, 3, 2, 4, 1, 3, 4, 1, 1, 2, 3, 3, 2, 2, 3, 3, 4, 2, 4, 1, 4, 1, 2],
                'blk2_rand2': [2, 1, 3, 3, 1, 4, 3, 1, 1, 4, 3, 1, 3, 4, 2, 1, 2, 3, 2, 2, 3, 4, 2, 4, 2, 4, 3, 2, 3, 2, 3, 3, 2, 2],
                'blk2_rand3': [4, 2, 3, 3, 3, 4, 4, 4, 1, 2, 4, 2, 2, 3, 1, 4, 3, 3, 4, 2, 3, 2, 1, 2, 1, 2, 3, 3, 1, 4, 1, 1, 2, 2],
                'blk2_rand4': [4, 4, 4, 4, 3, 2, 1, 4, 3, 4, 4, 4, 2, 3, 4, 1, 1, 1, 3, 3, 1, 4, 3, 1, 1, 4, 2, 3, 1, 1, 4, 3, 2, 2],
                'blk2_rand5': [1, 3, 1, 3, 1, 4, 2, 1, 3, 4, 1, 1, 1, 2, 1, 2, 2, 4, 1, 4, 1, 1, 3, 1, 2, 1, 3, 4, 1, 4, 3, 3, 1, 2],
                'blk3_rand0': [3, 4, 1, 3, 3, 4, 1, 1, 3, 2, 3, 3, 3, 3, 4, 1, 3, 2, 1, 2, 2, 2, 2, 4, 1, 1, 4, 2, 2, 1, 4, 1, 1, 3],
                'blk3_rand1': [1, 2, 4, 2, 2, 2, 2, 3, 1, 1, 4, 1, 1, 2, 1, 4, 3, 3, 1, 4, 1, 4, 4, 3, 3, 2, 3, 2, 2, 2, 2, 2, 1, 3],
                'blk3_rand2': [2, 1, 2, 1, 1, 1, 4, 2, 2, 2, 1, 2, 2, 1, 3, 1, 2, 1, 3, 1, 2, 3, 4, 2, 3, 3, 4, 1, 1, 2, 4, 1, 1, 3],
                'blk3_rand3': [3, 3, 4, 2, 3, 3, 2, 3, 1, 1, 2, 4, 2, 2, 1, 4, 3, 1, 2, 4, 3, 3, 1, 4, 2, 4, 4, 1, 3, 1, 1, 2, 2, 3],
                'blk3_rand4': [3, 2, 4, 2, 2, 1, 1, 2, 3, 1, 3, 2, 2, 4, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 3, 1, 2, 2, 1, 3, 1, 3, 2, 3],
                'blk3_rand5': [4, 2, 4, 1, 1, 1, 3, 2, 1, 1, 2, 1, 2, 2, 2, 4, 2, 4, 2, 3, 1, 3, 1, 4, 3, 2, 1, 1, 3, 4, 3, 4, 1, 3],
                'blk4_rand0': [3, 3, 2, 4, 3, 2, 2, 3, 2, 3, 4, 3, 1, 3, 1, 3, 3, 1, 1, 3, 2, 4, 1, 4, 2, 2, 2, 1, 2, 1, 2, 4, 3, 4],
                'blk4_rand1': [2, 1, 1, 2, 2, 2, 3, 4, 2, 3, 4, 1, 3, 2, 1, 1, 1, 3, 2, 1, 1, 1, 3, 3, 2, 4, 3, 1, 2, 1, 1, 3, 2, 4],
                'blk4_rand2': [2, 1, 4, 2, 2, 4, 3, 4, 3, 3, 4, 3, 1, 3, 2, 4, 1, 4, 2, 2, 2, 3, 3, 4, 1, 1, 4, 3, 2, 4, 1, 3, 3, 4],
                'blk4_rand3': [2, 1, 4, 1, 1, 3, 2, 4, 3, 1, 2, 3, 2, 3, 2, 1, 1, 4, 2, 3, 1, 4, 1, 4, 1, 4, 4, 2, 2, 4, 3, 3, 3, 4],
                'blk4_rand4': [3, 1, 1, 2, 3, 3, 1, 1, 2, 2, 3, 4, 3, 1, 3, 4, 2, 4, 4, 4, 2, 4, 4, 4, 1, 4, 2, 4, 2, 4, 2, 4, 2, 4],
                'blk4_rand5': [1, 4, 4, 3, 2, 2, 4, 4, 3, 4, 3, 2, 2, 1, 2, 4, 2, 3, 2, 1, 2, 3, 3, 4, 2, 3, 1, 4, 1, 2, 2, 2, 3, 4],
                'blk5_rand0': [1, 3, 1, 3, 3, 2, 2, 4, 1, 3, 3, 4, 3, 1, 4, 1, 2, 1, 3, 3, 1, 3, 3, 1, 1, 4, 3, 3, 3, 2, 4, 2, 2, 5],
                'blk5_rand1': [2, 2, 2, 4, 1, 4, 4, 2, 2, 4, 1, 3, 1, 2, 2, 1, 3, 2, 2, 2, 1, 4, 2, 4, 3, 2, 2, 3, 2, 4, 1, 4, 1, 5],
                'blk5_rand2': [3, 4, 1, 3, 2, 3, 4, 4, 3, 2, 3, 4, 1, 1, 3, 4, 1, 4, 3, 1, 1, 3, 4, 1, 3, 1, 1, 2, 2, 3, 2, 3, 2, 5],
                'blk5_rand3': [4, 1, 1, 4, 3, 2, 3, 1, 1, 1, 1, 4, 1, 4, 4, 1, 1, 2, 3, 1, 2, 2, 3, 4, 1, 3, 1, 2, 2, 4, 2, 2, 3, 5],
                'blk5_rand4': [3, 3, 3, 3, 2, 4, 1, 2, 1, 4, 3, 4, 2, 3, 2, 3, 1, 1, 1, 4, 2, 4, 1, 2, 1, 4, 4, 4, 1, 2, 2, 1, 2, 5],
                'blk5_rand5': [2, 3, 3, 4, 3, 3, 3, 1, 3, 3, 4, 3, 3, 2, 4, 4, 1, 3, 4, 3, 3, 1, 4, 2, 1, 2, 3, 3, 1, 1, 2, 4, 2, 5],
                'blk6_rand0': [3, 4, 1, 4, 3, 3, 2, 1, 2, 4, 4, 2, 2, 2, 2, 2, 2, 1, 3, 2, 2, 4, 2, 2, 2, 4, 2, 3, 3, 4, 2, 3, 1, 6],
                'blk6_rand1': [3, 2, 1, 4, 3, 1, 4, 4, 2, 1, 4, 3, 3, 2, 4, 1, 3, 4, 4, 2, 3, 3, 1, 3, 1, 3, 2, 3, 1, 1, 2, 3, 3, 6],
                'blk6_rand2': [4, 3, 3, 2, 1, 4, 1, 2, 1, 2, 4, 4, 2, 3, 2, 3, 1, 1, 1, 4, 1, 3, 1, 2, 2, 4, 2, 3, 1, 4, 1, 1, 3, 6],
                'blk6_rand3': [1, 2, 3, 1, 3, 1, 1, 1, 2, 2, 3, 1, 1, 2, 1, 2, 1, 3, 1, 1, 2, 1, 1, 4, 2, 4, 3, 4, 3, 2, 2, 2, 2, 6],
                'blk6_rand4': [3, 3, 2, 4, 1, 2, 4, 1, 2, 2, 3, 1, 1, 4, 3, 3, 3, 3, 1, 1, 2, 2, 3, 3, 3, 2, 4, 4, 2, 2, 2, 2, 2, 6],
                'blk6_rand5': [1, 3, 2, 1, 2, 2, 1, 3, 3, 4, 1, 4, 3, 1, 4, 1, 1, 3, 4, 4, 1, 2, 3, 3, 3, 3, 4, 4, 2, 3, 3, 1, 3, 6],
                'blk7_rand0': [2, 4, 1, 3, 3, 1, 1, 4, 2, 4, 3, 3, 1, 4, 2, 3, 3, 2, 2, 3, 3, 2, 3, 1, 1, 2, 4, 2, 2, 4, 4, 4, 1, 7],
                'blk7_rand1': [1, 3, 4, 1, 2, 1, 2, 4, 3, 1, 3, 1, 2, 3, 1, 2, 3, 3, 1, 1, 1, 2, 1, 3, 3, 1, 1, 4, 3, 2, 4, 1, 1, 7],
                'blk7_rand2': [1, 4, 3, 3, 3, 4, 2, 2, 2, 4, 3, 2, 2, 1, 2, 2, 3, 4, 3, 3, 3, 3, 3, 2, 3, 3, 4, 1, 2, 2, 2, 3, 3, 7],
                'blk7_rand3': [2, 4, 1, 2, 3, 3, 1, 2, 1, 1, 2, 2, 2, 3, 1, 1, 3, 2, 3, 3, 3, 1, 3, 1, 1, 1, 2, 4, 1, 1, 2, 1, 3, 7],
                'blk7_rand4': [3, 1, 1, 3, 1, 2, 4, 2, 1, 2, 1, 4, 3, 3, 2, 1, 1, 2, 3, 4, 2, 4, 3, 1, 3, 2, 1, 2, 2, 3, 2, 3, 1, 7],
                'blk7_rand5': [4, 1, 1, 4, 1, 2, 4, 3, 2, 2, 4, 4, 1, 2, 4, 4, 2, 4, 1, 2, 1, 1, 1, 1, 2, 1, 1, 4, 2, 2, 3, 2, 2, 7],
                'blk8_rand0': [2, 2, 4, 3, 3, 2, 3, 1, 2, 3, 2, 1, 2, 2, 2, 2, 1, 4, 2, 2, 3, 2, 3, 4, 1, 1, 4, 1, 1, 3, 4, 2, 1, 8],
                'blk8_rand1': [2, 4, 2, 4, 2, 3, 4, 3, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 3, 4, 2, 3, 2, 2, 2, 3, 4, 3, 1, 3, 3, 3, 3, 8],
                'blk8_rand2': [1, 1, 3, 4, 2, 1, 3, 3, 3, 2, 1, 4, 3, 2, 2, 2, 3, 1, 4, 3, 3, 1, 1, 1, 2, 1, 3, 1, 3, 4, 2, 4, 3, 8],
                'blk8_rand3': [2, 4, 4, 3, 2, 4, 3, 1, 2, 2, 2, 1, 1, 4, 1, 1, 1, 2, 4, 1, 2, 1, 3, 2, 3, 4, 4, 2, 3, 1, 1, 1, 2, 8],
                'blk8_rand4': [3, 3, 1, 2, 1, 1, 3, 4, 1, 3, 4, 2, 3, 1, 1, 3, 3, 1, 2, 4, 2, 3, 2, 1, 3, 4, 1, 1, 2, 3, 4, 4, 3, 8],
                'blk8_rand5': [4, 3, 4, 2, 1, 1, 3, 4, 1, 4, 2, 1, 3, 2, 3, 2, 1, 3, 3, 2, 3, 2, 2, 2, 2, 1, 2, 4, 3, 4, 1, 4, 3, 8],
            }
        elif configs.model.arch.q_layer_name in ['maxwell_s0']:
            arch_dict = {
                'blk1_rand0': [2, 2, 2, 2, 1, 3, 2, 4, 3, 3, 2, 1, 2, 1, 4, 3, 4, 1, 1, 4, 1, 4, 3, 4, 3, 4, 1, 4, 1, 4, 4, 2, 1, 2, 1, 2, 3, 4, 1, 1, 4, 1, 1, 2, 1],
                'blk1_rand1': [2, 3, 4, 1, 1, 1, 3, 3, 1, 4, 3, 1, 1, 4, 2, 3, 4, 1, 1, 4, 3, 2, 2, 1, 3, 2, 3, 3, 1, 2, 4, 1, 2, 2, 2, 2, 2, 3, 1, 3, 4, 2, 1, 2, 1],
                'blk1_rand2': [3, 1, 4, 4, 1, 1, 1, 1, 4, 2, 1, 1, 4, 4, 3, 1, 3, 4, 4, 2, 1, 3, 4, 4, 3, 2, 4, 1, 4, 1, 3, 1, 3, 1, 3, 1, 2, 1, 1, 2, 4, 4, 1, 2, 1],
                'blk1_rand3': [1, 4, 2, 3, 3, 2, 2, 3, 3, 2, 3, 1, 1, 2, 4, 2, 2, 4, 4, 4, 1, 3, 1, 3, 1, 3, 1, 4, 2, 1, 3, 4, 1, 1, 4, 1, 2, 1, 2, 2, 4, 1, 4, 1, 1],
                'blk1_rand4': [2, 4, 2, 4, 4, 1, 1, 4, 4, 3, 2, 3, 1, 1, 1, 1, 4, 1, 4, 4, 1, 1, 2, 3, 1, 2, 2, 3, 4, 1, 3, 1, 2, 2, 4, 2, 2, 3, 1, 1, 3, 4, 1, 2, 1],
                'blk1_rand5': [2, 4, 3, 1, 3, 1, 2, 3, 1, 2, 3, 3, 1, 1, 1, 2, 1, 3, 3, 1, 1, 4, 3, 2, 4, 1, 1, 3, 1, 2, 3, 1, 3, 1, 1, 1, 2, 2, 3, 1, 1, 2, 1, 2, 1],
                'blk1_rand6': [3, 4, 4, 1, 4, 3, 2, 1, 1, 2, 4, 2, 4, 2, 3, 4, 3, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 3, 4, 4, 2, 3, 2, 2, 2, 3, 4, 3, 1, 3, 3, 3, 3, 4, 1],
                'blk1_rand7': [4, 4, 3, 2, 2, 4, 4, 3, 4, 3, 2, 2, 1, 2, 4, 2, 3, 2, 1, 2, 3, 3, 4, 2, 3, 1, 4, 1, 2, 2, 2, 4, 3, 4, 1, 4, 1, 4, 3, 1, 2, 2, 4, 2, 1],
                'blk1_rand8': [3, 3, 2, 2, 3, 4, 2, 1, 4, 2, 2, 1, 1, 3, 4, 4, 3, 4, 4, 4, 4, 4, 1, 4, 1, 4, 3, 2, 1, 4, 3, 2, 1, 2, 3, 1, 2, 4, 1, 2, 3, 3, 1, 2, 1],
                'blk1_rand9': [1, 1, 1, 1, 2, 4, 4, 3, 2, 1, 2, 3, 3, 2, 2, 2, 4, 1, 2, 2, 2, 3, 3, 4, 1, 4, 1, 3, 3, 3, 4, 2, 2, 3, 2, 2, 4, 1, 3, 3, 4, 4, 2, 3, 1],
                'blk1_rand10': [4, 1, 1, 3, 3, 4, 3, 3, 3, 1, 4, 2, 1, 3, 1, 3, 3, 1, 4, 3, 4, 2, 3, 4, 4, 2, 4, 4, 2, 3, 4, 4, 4, 1, 2, 1, 2, 3, 3, 4, 2, 3, 2, 3, 1],
                'blk1_rand11': [3, 4, 4, 1, 2, 3, 1, 2, 1, 4, 2, 1, 1, 3, 1, 2, 2, 3, 2, 4, 2, 2, 1, 1, 2, 3, 1, 3, 2, 2, 4, 1, 2, 4, 1, 2, 2, 1, 1, 1, 2, 2, 4, 3, 1],
                'blk1_rand12': [1, 4, 4, 3, 2, 2, 2, 4, 3, 1, 4, 3, 3, 1, 1, 1, 2, 1, 3, 1, 3, 4, 2, 4, 4, 3, 4, 3, 1, 1, 3, 1, 2, 4, 2, 4, 1, 2, 1, 4, 3, 3, 2, 1, 1],
                'blk1_rand13': [1, 3, 4, 2, 3, 3, 4, 3, 3, 4, 1, 4, 1, 2, 1, 1, 2, 1, 4, 1, 2, 4, 2, 2, 3, 3, 2, 2, 2, 3, 3, 1, 1, 2, 4, 4, 3, 3, 3, 2, 4, 1, 2, 1, 1],
                'blk1_rand14': [2, 1, 4, 1, 4, 4, 4, 3, 1, 2, 2, 1, 3, 3, 4, 1, 1, 2, 1, 3, 1, 2, 4, 2, 2, 4, 2, 2, 1, 1, 4, 1, 1, 2, 3, 1, 3, 1, 1, 4, 4, 4, 3, 2, 1],
                'blk1_rand15': [1, 1, 3, 4, 2, 3, 4, 3, 2, 4, 3, 2, 1, 2, 1, 3, 4, 4, 3, 4, 4, 1, 3, 2, 1, 3, 1, 2, 2, 3, 2, 1, 2, 1, 1, 2, 3, 4, 2, 1, 1, 2, 3, 3, 1],
                'blk2_rand0': [4, 4, 3, 2, 2, 3, 2, 3, 4, 3, 4, 4, 1, 3, 1, 3, 3, 1, 1, 3, 2, 4, 1, 4, 2, 2, 2, 1, 2, 1, 2, 4, 4, 3, 4, 3, 4, 1, 4, 3, 3, 2, 1, 4, 2],
                'blk2_rand1': [1, 3, 2, 3, 1, 1, 2, 3, 3, 2, 3, 3, 1, 3, 3, 2, 2, 4, 1, 3, 3, 4, 3, 1, 4, 1, 4, 4, 2, 1, 3, 3, 1, 3, 3, 1, 4, 1, 4, 3, 3, 3, 2, 4, 2],
                'blk2_rand2': [4, 2, 4, 4, 2, 3, 4, 2, 1, 3, 1, 1, 2, 4, 2, 2, 2, 2, 3, 4, 1, 1, 4, 1, 4, 1, 2, 1, 4, 4, 4, 4, 4, 3, 3, 1, 4, 1, 4, 4, 3, 3, 2, 3, 2],
                'blk2_rand3': [3, 3, 4, 1, 4, 3, 1, 1, 4, 3, 1, 4, 3, 4, 2, 1, 2, 3, 2, 2, 3, 4, 2, 4, 2, 4, 3, 2, 3, 2, 3, 3, 2, 2, 2, 1, 4, 1, 1, 3, 2, 4, 3, 1, 2],
                'blk2_rand4': [3, 2, 3, 2, 1, 4, 1, 4, 2, 3, 1, 4, 1, 4, 1, 4, 4, 2, 4, 2, 4, 3, 3, 3, 4, 2, 2, 4, 3, 3, 2, 3, 1, 2, 3, 2, 1, 2, 2, 2, 2, 1, 4, 2, 2],
                'blk2_rand5': [3, 1, 1, 4, 2, 1, 1, 4, 4, 2, 4, 3, 4, 3, 2, 2, 2, 2, 2, 3, 3, 2, 4, 1, 2, 4, 1, 2, 2, 3, 1, 1, 4, 3, 3, 3, 3, 1, 1, 2, 2, 3, 3, 3, 2],
                'blk2_rand6': [4, 3, 3, 3, 1, 3, 3, 4, 3, 3, 2, 4, 4, 4, 4, 4, 4, 1, 3, 4, 3, 3, 1, 4, 2, 1, 2, 3, 3, 1, 1, 2, 4, 2, 1, 2, 1, 2, 1, 1, 1, 4, 2, 2, 2],
                'blk2_rand7': [3, 3, 3, 3, 2, 3, 3, 4, 1, 2, 2, 2, 3, 3, 3, 4, 1, 3, 1, 4, 2, 3, 1, 1, 3, 2, 2, 2, 4, 1, 1, 1, 4, 1, 4, 2, 3, 1, 1, 2, 3, 4, 3, 2, 2],
                'blk2_rand8': [2, 4, 1, 3, 4, 2, 1, 4, 1, 1, 1, 3, 3, 2, 2, 4, 2, 1, 1, 3, 1, 4, 1, 1, 4, 2, 2, 4, 3, 1, 4, 1, 3, 1, 4, 1, 2, 3, 3, 3, 4, 3, 2, 4, 2],
                'blk2_rand9': [3, 1, 3, 4, 1, 3, 4, 1, 4, 2, 2, 1, 3, 1, 4, 3, 2, 3, 2, 3, 3, 3, 4, 4, 1, 1, 2, 4, 2, 1, 2, 1, 4, 3, 4, 2, 2, 4, 4, 1, 2, 2, 4, 3, 2],
                'blk2_rand10': [4, 3, 2, 2, 3, 3, 4, 2, 2, 1, 4, 1, 2, 4, 3, 4, 3, 3, 4, 3, 1, 2, 3, 1, 3, 2, 1, 2, 2, 1, 1, 1, 4, 1, 1, 1, 1, 3, 4, 2, 1, 3, 3, 3, 2],
                'blk2_rand11': [1, 2, 4, 4, 3, 4, 1, 1, 4, 1, 2, 2, 4, 2, 2, 3, 3, 2, 4, 3, 1, 1, 1, 4, 2, 1, 4, 3, 2, 3, 4, 1, 1, 4, 1, 2, 4, 3, 4, 2, 2, 4, 4, 1, 2],
                'blk2_rand12': [1, 3, 4, 3, 4, 2, 2, 2, 3, 2, 1, 4, 4, 1, 3, 3, 4, 2, 3, 3, 1, 1, 2, 3, 3, 1, 3, 4, 1, 2, 1, 3, 2, 1, 4, 4, 1, 4, 4, 4, 2, 4, 4, 4, 2],
                'blk2_rand13': [2, 1, 1, 3, 4, 1, 4, 2, 1, 3, 2, 3, 2, 1, 3, 3, 2, 3, 2, 2, 2, 2, 1, 2, 4, 3, 4, 1, 4, 3, 4, 1, 2, 4, 1, 1, 3, 3, 4, 2, 1, 4, 1, 1, 2],
                'blk2_rand14': [4, 2, 4, 2, 2, 2, 1, 1, 4, 2, 1, 2, 3, 3, 3, 4, 2, 3, 1, 3, 3, 3, 4, 1, 2, 1, 2, 4, 4, 4, 4, 2, 3, 2, 1, 4, 4, 1, 2, 3, 3, 1, 3, 2, 2],
                'blk2_rand15': [1, 2, 4, 1, 3, 3, 4, 3, 4, 3, 2, 1, 2, 1, 1, 1, 2, 1, 2, 4, 3, 3, 1, 2, 3, 2, 2, 1, 3, 1, 1, 3, 3, 4, 4, 2, 3, 4, 2, 2, 2, 4, 3, 1, 2],
                'blk3_rand0': [4, 4, 2, 2, 2, 2, 2, 4, 2, 1, 3, 2, 2, 4, 2, 2, 2, 4, 2, 3, 4, 3, 4, 2, 3, 4, 1, 2, 4, 1, 4, 1, 2, 3, 1, 4, 2, 1, 4, 4, 4, 1, 1, 1, 3],
                'blk3_rand1': [1, 1, 1, 3, 1, 4, 1, 4, 4, 4, 3, 3, 3, 1, 4, 3, 3, 1, 3, 1, 2, 3, 2, 1, 4, 3, 1, 4, 4, 2, 1, 4, 3, 3, 2, 4, 1, 3, 4, 4, 2, 3, 3, 1, 3],
                'blk3_rand2': [2, 1, 2, 1, 1, 2, 4, 4, 4, 4, 4, 2, 2, 3, 4, 2, 3, 4, 1, 3, 2, 1, 1, 1, 3, 2, 1, 4, 1, 1, 3, 3, 2, 4, 3, 1, 2, 1, 1, 3, 2, 4, 4, 3, 3],
                'blk3_rand3': [2, 4, 4, 4, 1, 4, 1, 2, 1, 2, 4, 4, 2, 3, 2, 3, 1, 1, 1, 4, 1, 3, 1, 2, 2, 4, 2, 3, 1, 4, 1, 1, 3, 2, 2, 1, 4, 2, 4, 2, 4, 3, 4, 3, 3],
                'blk3_rand4': [4, 3, 1, 3, 2, 4, 1, 4, 2, 2, 2, 3, 3, 4, 4, 1, 1, 4, 3, 2, 4, 1, 3, 4, 4, 3, 4, 3, 2, 3, 3, 3, 4, 4, 3, 4, 1, 1, 4, 4, 3, 2, 4, 1, 3],
                'blk3_rand5': [3, 2, 3, 4, 1, 1, 4, 1, 4, 4, 4, 1, 3, 4, 2, 1, 4, 4, 2, 3, 3, 3, 4, 4, 4, 1, 2, 4, 2, 2, 3, 1, 4, 3, 3, 4, 2, 3, 2, 1, 2, 1, 2, 3, 3],
                'blk3_rand6': [3, 3, 1, 4, 4, 1, 1, 3, 4, 1, 3, 4, 2, 3, 4, 4, 3, 2, 3, 4, 1, 1, 3, 4, 1, 4, 3, 1, 4, 1, 3, 4, 1, 3, 1, 1, 2, 2, 3, 2, 3, 2, 1, 3, 3],
                'blk3_rand7': [1, 3, 3, 2, 3, 4, 1, 2, 2, 4, 1, 1, 2, 3, 2, 1, 4, 2, 1, 2, 3, 2, 2, 1, 1, 1, 2, 3, 4, 4, 1, 1, 2, 4, 1, 3, 3, 1, 1, 4, 4, 2, 4, 3, 3],
                'blk3_rand8': [4, 4, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 4, 1, 2, 1, 4, 3, 4, 2, 3, 2, 3, 1, 1, 1, 4, 2, 4, 1, 2, 1, 4, 4, 4, 1, 2, 2, 1, 2, 1, 2, 3, 3],
                'blk3_rand9': [1, 2, 2, 1, 3, 1, 2, 1, 3, 1, 2, 3, 4, 2, 3, 3, 4, 1, 1, 2, 4, 1, 4, 1, 3, 1, 4, 3, 3, 3, 4, 2, 2, 2, 4, 3, 2, 4, 2, 1, 2, 2, 3, 4, 3],
                'blk3_rand10': [2, 3, 3, 4, 1, 3, 3, 1, 2, 2, 4, 2, 1, 3, 2, 4, 2, 1, 4, 2, 4, 1, 3, 1, 1, 1, 1, 3, 3, 4, 2, 3, 3, 2, 3, 1, 1, 2, 4, 4, 2, 2, 1, 4, 3],
                'blk3_rand11': [1, 2, 4, 3, 3, 1, 4, 4, 2, 4, 4, 1, 4, 3, 1, 1, 2, 4, 2, 3, 1, 1, 1, 3, 3, 2, 2, 2, 4, 1, 1, 2, 1, 2, 4, 4, 3, 4, 2, 4, 3, 1, 2, 1, 3],
                'blk3_rand12': [1, 2, 2, 4, 2, 3, 1, 1, 3, 2, 3, 3, 4, 3, 1, 3, 1, 1, 1, 2, 4, 1, 1, 2, 1, 3, 3, 4, 1, 2, 3, 2, 2, 3, 2, 3, 2, 3, 1, 4, 3, 3, 4, 3, 3],
                'blk3_rand13': [2, 2, 1, 3, 1, 3, 2, 3, 4, 2, 4, 1, 1, 1, 3, 2, 1, 1, 2, 1, 4, 2, 2, 2, 4, 2, 4, 2, 3, 1, 3, 1, 4, 3, 2, 1, 1, 3, 4, 3, 4, 4, 1, 3, 3],
                'blk3_rand14': [2, 3, 4, 2, 4, 3, 1, 3, 2, 1, 2, 2, 3, 2, 3, 1, 3, 1, 3, 4, 4, 2, 3, 3, 1, 2, 3, 2, 4, 3, 3, 2, 3, 2, 3, 1, 2, 3, 4, 3, 4, 4, 1, 2, 3],
                'blk3_rand15': [2, 2, 4, 3, 2, 2, 2, 4, 3, 1, 4, 4, 3, 3, 4, 1, 4, 3, 1, 3, 1, 2, 3, 3, 4, 3, 3, 2, 3, 2, 2, 2, 2, 1, 4, 1, 3, 1, 1, 3, 3, 3, 1, 1, 3],
                'blk4_rand0': [3, 4, 1, 3, 3, 4, 1, 1, 3, 2, 3, 3, 3, 3, 4, 1, 4, 4, 4, 3, 2, 1, 2, 4, 4, 2, 2, 2, 4, 4, 1, 1, 4, 2, 2, 1, 4, 1, 1, 3, 3, 3, 2, 4, 4],
                'blk4_rand1': [4, 1, 1, 2, 3, 3, 2, 2, 3, 3, 4, 4, 2, 4, 1, 4, 4, 1, 2, 1, 4, 2, 4, 1, 1, 4, 3, 3, 1, 1, 3, 3, 4, 4, 3, 4, 4, 2, 4, 1, 4, 4, 3, 1, 4],
                'blk4_rand2': [1, 4, 1, 1, 2, 2, 1, 3, 2, 4, 2, 2, 4, 1, 4, 1, 1, 1, 4, 1, 3, 4, 3, 1, 2, 1, 3, 3, 4, 2, 4, 3, 1, 2, 1, 1, 2, 4, 1, 3, 1, 2, 2, 2, 4],
                'blk4_rand3': [1, 4, 4, 2, 2, 4, 1, 3, 4, 1, 2, 2, 1, 3, 2, 2, 2, 1, 4, 2, 4, 3, 2, 2, 3, 2, 4, 1, 4, 1, 1, 4, 4, 4, 4, 3, 2, 1, 4, 4, 4, 3, 4, 4, 4],
                'blk4_rand4': [3, 1, 4, 2, 1, 3, 4, 1, 4, 3, 3, 4, 1, 2, 3, 1, 1, 2, 3, 3, 1, 1, 2, 2, 3, 4, 3, 1, 3, 4, 2, 4, 4, 4, 2, 4, 4, 4, 1, 4, 2, 4, 4, 2, 4],
                'blk4_rand5': [3, 3, 1, 1, 1, 1, 3, 3, 2, 2, 1, 4, 4, 3, 3, 3, 2, 2, 3, 3, 3, 4, 2, 4, 4, 3, 3, 1, 1, 4, 2, 3, 3, 4, 1, 2, 1, 1, 1, 1, 3, 3, 1, 2, 4],
                'blk4_rand6': [2, 2, 2, 1, 3, 2, 1, 2, 2, 1, 3, 3, 4, 1, 4, 4, 4, 3, 1, 4, 1, 1, 3, 4, 4, 1, 2, 3, 3, 3, 3, 4, 4, 2, 3, 3, 1, 4, 3, 2, 2, 4, 2, 2, 4],
                'blk4_rand7': [1, 3, 3, 2, 4, 4, 3, 2, 4, 3, 1, 4, 2, 2, 2, 1, 4, 4, 1, 4, 1, 1, 1, 2, 4, 1, 4, 2, 1, 3, 2, 3, 4, 4, 2, 4, 3, 1, 1, 1, 2, 4, 3, 4, 4],
                'blk4_rand8': [4, 4, 2, 3, 2, 4, 2, 3, 3, 3, 1, 2, 4, 4, 4, 1, 1, 3, 4, 1, 3, 4, 2, 3, 1, 1, 3, 3, 1, 2, 4, 2, 3, 2, 1, 3, 4, 1, 1, 4, 4, 2, 3, 4, 4],
                'blk4_rand9': [4, 4, 2, 4, 1, 2, 1, 1, 1, 1, 2, 1, 1, 4, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 3, 4, 1, 4, 3, 1, 3, 3, 2, 1, 2, 1, 3, 3, 4, 1, 2, 4],
                'blk4_rand10': [2, 1, 1, 2, 2, 4, 1, 3, 2, 4, 3, 1, 4, 2, 3, 2, 1, 3, 1, 1, 4, 2, 3, 1, 1, 3, 1, 4, 4, 4, 4, 1, 3, 3, 3, 3, 2, 3, 2, 2, 4, 2, 1, 2, 4],
                'blk4_rand11': [3, 2, 1, 1, 3, 1, 1, 1, 3, 2, 4, 2, 4, 1, 2, 4, 3, 3, 2, 2, 1, 2, 1, 1, 3, 3, 4, 3, 3, 2, 3, 4, 2, 4, 3, 4, 2, 1, 4, 4, 2, 2, 4, 3, 4],
                'blk4_rand12': [4, 2, 3, 3, 1, 3, 4, 1, 1, 1, 1, 4, 1, 3, 4, 2, 2, 3, 3, 4, 2, 2, 2, 2, 2, 1, 3, 1, 2, 4, 4, 2, 2, 4, 1, 1, 2, 1, 4, 1, 2, 4, 1, 3, 4],
                'blk4_rand13': [4, 3, 3, 4, 3, 2, 2, 2, 4, 3, 3, 1, 1, 3, 3, 3, 4, 3, 1, 4, 3, 1, 4, 3, 3, 3, 4, 1, 1, 2, 3, 2, 2, 1, 4, 4, 1, 4, 2, 3, 1, 3, 4, 3, 4],
                'blk4_rand14': [4, 2, 2, 2, 4, 3, 4, 1, 1, 4, 4, 2, 4, 2, 1, 2, 4, 1, 3, 2, 1, 2, 1, 1, 2, 3, 1, 1, 1, 3, 3, 4, 2, 3, 2, 4, 2, 3, 2, 3, 2, 3, 3, 2, 4],
                'blk4_rand15': [3, 3, 3, 1, 3, 2, 2, 1, 2, 2, 3, 2, 1, 3, 4, 3, 1, 3, 3, 1, 3, 1, 1, 4, 2, 2, 3, 3, 4, 4, 4, 1, 4, 4, 3, 4, 1, 1, 3, 4, 4, 1, 4, 4, 4],
            }

        sample_arch = arch_dict[name.replace('sharefront0_', '')]
    else:
        raise NotImplementedError(f"Arch name {name} not supported.")

    return sample_arch
