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

from torchpack.datasets.dataset import Dataset
import numpy as np
import torch

__all__ = ["LayerRegression"]


class LayerRegressionDatasetV:
    def __init__(self):
        self.data = None
        self.n_instance = 10000
        self.input = np.random.rand(5).astype(np.float) * 2 * np.pi
        self.output = np.random.rand(5).astype(np.float) * 2 - 1

    def __getitem__(self, index: int):
        instance = {"input": self.input, "output": self.output}
        return instance

    def __len__(self) -> int:
        return self.n_instance


class LayerRegressionV(Dataset):
    def __init__(
        self,
    ):
        super().__init__(
            {split: LayerRegressionDatasetV() for split in ["train", "valid", "test"]}
        )


class LayerRegressionDataset:
    def __init__(self):
        self.data = None
        self.n_instance = 10000
        mat = torch.randn((2**5, 2**5), dtype=torch.complex64)
        mat = mat.svd()[0].data

        self.output = mat

    def __getitem__(self, index: int):
        instance = {"input": self.output, "output": self.output}
        return instance

    def __len__(self) -> int:
        return self.n_instance


class LayerRegression(Dataset):
    def __init__(
        self,
    ):
        super().__init__(
            {split: LayerRegressionDataset() for split in ["train", "valid", "test"]}
        )
