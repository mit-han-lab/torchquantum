import torch

from torchpack.datasets.dataset import Dataset
from torchvision import datasets, transforms
from typing import List

__all__ = ['MNIST']


class MNISTDataset:
    def __init__(self,
                 root: str,
                 split: str,
                 train_valid_split_ratio: List[float],
                 center_crop,
                 resize,
                 binarize,
                 binarize_threshold,
                 digits_of_interest):
        self.root = root
        self.split = split
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.center_crop = center_crop
        self.resize = resize
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.digits_of_interest = digits_of_interest

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        tran = [transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]
        if not self.center_crop == 28:
            tran.append(transforms.CenterCrop(self.center_crop))
        if not self.resize == 28:
            tran.append(transforms.Resize(self.resize))
        transform = transforms.Compose(tran)

        if self.split == 'train' or self.split == 'valid':
            train_valid = datasets.MNIST(self.root, train=True,
                                         download=True, transform=transform)
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
            test = datasets.MNIST(self.root, train=False, transform=transform)
            idx, _ = torch.stack([test.targets == number for number in
                                  self.digits_of_interest]).max(dim=0)
            test.targets = test.targets[idx]
            test.data = test.data[idx]

            self.data = test

    def __getitem__(self, index: int):
        img = self.data[index][0]
        if self.binarize:
            img = 1. * (img > self.binarize_threshold) + \
                  -1. * (img <= self.binarize_threshold)
        instance = {'image': img, 'digit': self.data[index][1]}
        return instance

    def __len__(self) -> int:
        return self.n_instance


class MNIST(Dataset):
    def __init__(self,
                 root: str,
                 train_valid_split_ratio: List[float],
                 center_crop=28,
                 resize=28,
                 binarize=False,
                 binarize_threshold=0.1307,
                 digits_of_interest=tuple(range(10)),
                 ):
        self.root = root

        super().__init__({
            split: MNISTDataset(
                root=root,
                split=split,
                train_valid_split_ratio=train_valid_split_ratio,
                center_crop=center_crop,
                resize=resize,
                binarize=binarize,
                binarize_threshold=binarize_threshold,
                digits_of_interest=digits_of_interest
            )
            for split in ['train', 'valid', 'test']
        })


if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    mnist = MNISTDataset(root='../data',
                         split='train',
                         train_valid_split_ratio=[0.9, 0.1],
                         center_crop=28,
                         resize=28,
                         binarize=False,
                         binarize_threshold=0.1307,
                         digits_of_interest=(3, 6)
                         )
    print('finish')
