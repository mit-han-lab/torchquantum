import torch

from torchpack.datasets.dataset import Dataset
from torchvision import datasets, transforms
from typing import List

__all__ = ['MNISTBinarized']


class MNISTBinarizedDataset:
    def __init__(self,
                 root: str,
                 split: str,
                 train_valid_split_ratio: List[float],
                 threshold: float):
        self.root = root
        self.split = split
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.threshold = threshold

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(20),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize(4)
        ])

        if self.split == 'train' or self.split == 'valid':
            train_valid = datasets.MNIST(self.root, train=True,
                                         download=True, transform=transform)
            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            split = [train_len, len(train_valid) - train_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1))

            if self.split == 'train':
                self.data = train_subset
            else:
                self.data = valid_subset

        else:
            test = datasets.MNIST(self.root, train=False, transform=transform)
            self.data = test

    def __getitem__(self, index: int):
        img = self.data[index][0]
        img = 1. * (img > self.threshold) + -1. * (img <= self.threshold)

        instance = {'image': img, 'digit': self.data[index][1]}
        return instance

    def __len__(self) -> int:
        return self.n_instance


class MNISTBinarized(Dataset):
    def __init__(self,
                 root: str,
                 train_valid_split_ratio: List[float],
                 threshold: float = 0.1307
                 ):
        self.root = root

        super().__init__({
            split: MNISTBinarizedDataset(
                root=root,
                split=split,
                train_valid_split_ratio=train_valid_split_ratio,
                threshold=threshold
            )
            for split in ['train', 'valid', 'test']
        })


if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    mnist = MNISTBinarizedDataset(root='../data',
                                  split='train',
                                  train_valid_split_ratio=[0.9, 0.1],
                                  threshold=0.1307
                                  )
    mnist.__getitem__(1)
    print('finish')