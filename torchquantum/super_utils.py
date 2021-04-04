import numpy as np
import torch.nn as nn
from typing import Iterable
from torchpack.utils.logging import logger


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
    else:
        raise NotImplementedError(f"Arch name {name} not supported.")

    return sample_arch
