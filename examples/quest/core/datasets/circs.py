"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import pdb
import pickle
import random
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy
import scipy.signal
import torch
from torchpack.datasets.dataset import Dataset

from utils.circ_dag_converter import circ_to_dag_with_data
from utils.load_data import load_data_from_pyg

__all__ = ["CircDataset", "Circ"]


class CircDataset:
    def __init__(self, root: str, split_ratio: List[float]):
        super().__init__()
        self.root = root
        self.split_ratio = split_ratio
        self.raw = {}
        self.mean = {}
        self.std = {}

        self._load()
        self._preprocess()
        self._split()

        self.instance_num = len(self.raw["pygdataset"])

    def _load(self):
        self.raw["pygdataset"] = load_data_from_pyg(self.root)

        # shuffle features
        # random.seed(42)
        # np.random.seed(42)
        # shuffler = np.random.permutation(len(self.raw['pygdataset']))
        # for feat_name in self.raw.keys():
        #     random.shuffle(self.raw[feat_name])

    def _preprocess(self):
        pass
        # compute the mean and std of each feature:
        # all_features = None
        # for k, dag in enumerate(self.raw["pygdataset"]):
        #     if not k:
        #         all_features = dag.x
        #         global_features = dag.global_features
        #     else:
        #         all_features = torch.cat([all_features, dag.x])
        #         global_features = torch.cat([global_features, dag.global_features])
        # self.means = all_features.mean(0)
        # self.stds = all_features.std(0)
        # self.means_gf = global_features.mean(0)
        # self.stds_gf = global_features.std(0)
        # for k, dag in enumerate(self.raw["pygdataset"]):
        #     self.raw["pygdataset"][k].x = (dag.x - self.means) / (1e-8 + self.stds)
        #     self.raw["pygdataset"][k].gf = (dag.global_features - self.means_gf) / (
        #         1e-8 + self.stds_gf
        #     )

    def _split(self):
        instance_num = len(self.raw["pygdataset"])
        split_train = self.split_ratio[0]
        split_valid = self.split_ratio[0] + self.split_ratio[1]

        self.raw["train"] = self.raw["pygdataset"][: int(split_train * instance_num)]
        self.raw["valid"] = self.raw["pygdataset"][
            int(split_train * instance_num) : int(split_valid * instance_num)
        ]
        self.raw["test"] = self.raw["pygdataset"][int(split_valid * instance_num) :]

    def get_data(self, device, split):
        return [data.to(device) for data in self.raw[split]]

    def __getitem__(self, index: int):
        data_this = {"dag": self.raw["pygdataset"][index]}
        return data_this

    def __len__(self) -> int:
        return self.instance_num


class Circ(Dataset):
    def __init__(
        self,
        root: str,
        split_ratio: List[float],
        resample_len=None,
        location=None,
        augment_setting=None,
    ):
        self.root = root

        super().__init__(
            {
                split: CircDataset(
                    root=root,
                    split=split,
                    split_ratio=split_ratio,
                )
                for split in ["train", "valid", "test"]
            }
        )


if __name__ == "__main__":
    # import matplotlib
    # matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    pdb.set_trace()
    circ_data = CircDataset(
        root="./data/1.data.dags", split="train", split_ratio=[0.8, 0.1, 0.1]
    )
    # plt.plot(pd.raw['a'][0])
    # plt.show()
    print(circ_data[0])
    print("Finish")
