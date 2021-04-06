"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-04-04 13:38:43
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-04-04 15:16:55
"""

import os
import numpy as np
import torch

from torch import Tensor
from torchpack.datasets.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url
from typing import Any, Callable, Dict, List, Optional, Tuple


__all__ = ["Vowel"]


class VowelRecognition(VisionDataset):
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases" \
          "/undocumented/connectionist-bench/vowel/vowel-context.data"
    filename = "vowel-context.data"
    folder = "vowel"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        n_features: int = 10,
        train_ratio: float = 0.7,
        download: bool = False
    ) -> None:
        root = os.path.join(os.path.expanduser(root), self.folder)
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        super(VowelRecognition, self).__init__(
            root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.train_ratio = train_ratio

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.n_features = n_features
        assert 1 <= n_features <= 10, print(
            f"Only support maximum 10 features, but got{n_features}")
        self.data: Any = []
        self.targets = []

        self.process_raw_data()
        self.data, self.targets = self.load(train=train)

    def process_raw_data(self) -> None:
        processed_dir = os.path.join(self.root, "processed")
        processed_training_file = os.path.join(processed_dir, "training.pt")
        processed_test_file = os.path.join(processed_dir, "test.pt")
        if os.path.exists(processed_training_file) and \
                os.path.exists(processed_test_file):
            with open(os.path.join(self.root, "processed/training.pt"),
                      'rb') as f:
                data, targets = torch.load(f)
                if data.shape[-1] == self.n_features:
                    print('Data already processed')
                    return
        data, targets = self._load_dataset()
        data_train, targets_train, data_test, targets_test = \
            self._split_dataset(data, targets)
        data_train, data_test = self._preprocess_dataset(data_train, data_test)
        self._save_dataset(data_train, targets_train,
                           data_test, targets_test, processed_dir)

    def _load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        data = []
        targets = []
        with open(os.path.join(self.root, "raw", self.filename), 'r') as f:
            for line in f:
                line = line.strip().split()[3:]
                label = int(line[-1])
                targets.append(label)
                example = [float(i) for i in line[:-1]]
                data.append(example)

            data = torch.Tensor(data)
            targets = torch.LongTensor(targets)
        return data, targets

    def _split_dataset(self,
                       data: Tensor,
                       targets: Tensor) -> Tuple[Tensor, ...]:
        from sklearn.model_selection import train_test_split
        data_train, data_test, targets_train, targets_test = train_test_split(
            data, targets, train_size=self.train_ratio, random_state=42)
        print(
            f'training: {data_train.shape[0]} examples, '
            f'test: {data_test.shape[0]} examples')
        return data_train, targets_train, data_test, targets_test

    def _preprocess_dataset(self,
                            data_train: Tensor,
                            data_test: Tensor) -> Tuple[Tensor, Tensor]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler, RobustScaler
        pca = PCA(n_components=self.n_features)
        data_train_reduced = pca.fit_transform(data_train)
        data_test_reduced = pca.transform(data_test)

        rs = RobustScaler(quantile_range=(10, 90)).fit(
            np.concatenate([data_train_reduced, data_test_reduced], 0))
        data_train_reduced = rs.transform(data_train_reduced)
        data_test_reduced = rs.transform(data_test_reduced)
        mms = MinMaxScaler()
        mms.fit(np.concatenate([data_train_reduced, data_test_reduced], 0))
        data_train_reduced = mms.transform(data_train_reduced)
        data_test_reduced = mms.transform(data_test_reduced)

        return torch.from_numpy(data_train_reduced).float(), \
            torch.from_numpy(data_test_reduced).float()

    @staticmethod
    def _save_dataset(data_train: Tensor,
                      targets_train: Tensor,
                      data_test: Tensor,
                      targets_test: Tensor,
                      processed_dir: str) -> None:
        os.makedirs(processed_dir, exist_ok=True)
        processed_training_file = os.path.join(processed_dir, "training.pt")
        processed_test_file = os.path.join(processed_dir, "test.pt")
        with open(processed_training_file, 'wb') as f:
            torch.save((data_train, targets_train), f)

        with open(processed_test_file, 'wb') as f:
            torch.save((data_test, targets_test), f)
        print(f'Processed dataset saved')

    def load(self, train: bool = True):
        filename = "training.pt" if train else "test.pt"
        with open(os.path.join(self.root, "processed", filename), 'rb') as f:
            data, targets = torch.load(f)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
        return data, targets

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_url(self.url, root=os.path.join(
            self.root, "raw"), filename=self.filename)

    def _check_integrity(self) -> bool:
        return os.path.exists(os.path.join(self.root, "raw", self.filename))

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class VowelRecognitionDataset:
    def __init__(self,
                 root: str,
                 split: str,
                 test_ratio: float,
                 train_valid_split_ratio: List[float],
                 resize: int,
                 binarize: bool,
                 binarize_threshold: float,
                 digits_of_interest: List[int]):
        self.root = root
        self.split = split
        self.test_ratio = test_ratio
        assert 0 < test_ratio < 1, print(
            f"Only support test_ratio from (0, 1), but got {test_ratio}")
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.resize = resize
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.digits_of_interest = digits_of_interest

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        tran = [transforms.ToTensor()]
        transform = transforms.Compose(tran)

        if self.split == 'train' or self.split == 'valid':
            train_valid = VowelRecognition(self.root, train=True,
                                           download=True, transform=transform,
                                           n_features=self.resize,
                                           train_ratio=1-self.test_ratio)
            idx, _ = torch.stack([train_valid.targets == number for number in
                                  self.digits_of_interest]).max(dim=0)
            train_valid.targets = train_valid.targets[idx]
            train_valid.data = train_valid.data[idx]

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            split = [train_len, len(train_valid) - train_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1))

            if self.split == 'train':
                self.data = train_subset
            else:
                self.data = valid_subset

        else:
            test = VowelRecognition(self.root, train=False,
                                    download=True, transform=transform,
                                    n_features=self.resize,
                                    train_ratio=1-self.test_ratio)
            idx, _ = torch.stack([test.targets == number for number in
                                  self.digits_of_interest]).max(dim=0)
            test.targets = test.targets[idx]
            test.data = test.data[idx]

            self.data = test

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        data = self.data[index][0]
        if self.binarize:
            data = 1. * (data > self.binarize_threshold) + \
                -1. * (data <= self.binarize_threshold)

        digit = self.digits_of_interest.index(self.data[index][1])
        instance = {'vowel': data, 'digit': digit}
        return instance

    def __len__(self) -> int:
        return len(self.data)

    def __call__(self, index: int) -> Dict[str, Tensor]:
        return self.__getitem__(index)


class Vowel(Dataset):
    def __init__(self,
                 root: str,
                 test_ratio: float,
                 train_valid_split_ratio: List[float],
                 resize=28,
                 binarize=False,
                 binarize_threshold=0.1307,
                 digits_of_interest=tuple(range(10)),
                 ):
        self.root = root

        super().__init__({
            split: VowelRecognitionDataset(
                root=root,
                split=split,
                test_ratio=test_ratio,
                train_valid_split_ratio=train_valid_split_ratio,
                resize=resize,
                binarize=binarize,
                binarize_threshold=binarize_threshold,
                digits_of_interest=digits_of_interest
            )
            for split in ['train', 'valid', 'test']
        })


def test_vowel():
    import pdb
    pdb.set_trace()
    vowel = VowelRecognition(root=".", download=True, n_features=6)
    print(vowel.data.size(), vowel.targets.size())
    vowel = VowelRecognitionDataset(root=".", split="train", test_ratio=0.3,
                                    train_valid_split_ratio=[0.9, 0.1],
                                    resize=8,
                                    binarize=0,
                                    binarize_threshold=0,
                                    digits_of_interest=tuple(range(10)))
                                    # digits_of_interest=(3, 6))
    print(vowel(20))


if __name__ == "__main__":
    test_vowel()
