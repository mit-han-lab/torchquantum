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


__all__ = ["VQE"]


class VQEDataset:
    def __init__(self, split, steps_per_epoch):
        self.split = split
        self.steps_per_epoch = steps_per_epoch

    def __getitem__(self, index: int):
        instance = {"input": -1, "target": -1}

        return instance

    def __len__(self) -> int:
        if self.split == "train":
            return self.steps_per_epoch
        else:
            return 1


class VQE(Dataset):
    def __init__(self, steps_per_epoch):
        super().__init__(
            {
                split: VQEDataset(split=split, steps_per_epoch=steps_per_epoch)
                for split in ["train", "valid", "test"]
            }
        )
