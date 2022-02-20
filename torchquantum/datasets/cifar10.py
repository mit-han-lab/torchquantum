import torch

from torchpack.datasets.dataset import Dataset
from torchvision import datasets, transforms
from typing import List
from torchpack.utils.logging import logger
from torchvision.transforms import InterpolationMode


__all__ = ['CIFAR10']


resize_modes = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'nearest': InterpolationMode.NEAREST,
}


class CIFAR10Dataset:
    def __init__(self,
                 root: str,
                 split: str,
                 train_valid_split_ratio: List[float],
                 center_crop,
                 resize,
                 resize_mode,
                 binarize,
                 binarize_threshold,
                 grayscale,
                 digits_of_interest,
                 n_test_samples,
                 n_valid_samples,
                 fashion,
                 ):
        self.root = root
        self.split = split
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.center_crop = center_crop
        self.resize = resize
        self.resize_mode = resize_modes[resize_mode]
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.grayscale = grayscale
        self.digits_of_interest = digits_of_interest
        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples
        self.fashion = fashion

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        if self.grayscale:
            tran = [transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Normalize((0.2989*0.4914+0.587*0.4822+0.114*0.4465, ), (((0.2989*0.2023)**2+(0.587*0.1994)**2+(0.114*0.2010)**2)**0.5,))
                    ]
        else:
            tran = [transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ]


        if not self.center_crop == 32:
            tran.append(transforms.CenterCrop(self.center_crop))
        if not self.resize == 32:
            tran.append(transforms.Resize(self.resize,
                                          interpolation=self.resize_mode))
        transform = transforms.Compose(tran)

        if self.split == 'train' or self.split == 'valid':
            train_valid = datasets.CIFAR10(
                    self.root, train=True, download=True, transform=transform)
            targets = torch.tensor(train_valid.targets)
            idx, _ = torch.stack([targets == number for number in
                                  self.digits_of_interest]).max(dim=0)
            # targets = targets[idx]
            train_valid.targets = targets[idx].numpy().tolist()
            train_valid.data = train_valid.data[idx]

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            split = [train_len, len(train_valid) - train_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1))
            if self.split == 'train':
                self.data = train_subset
            else:
                if self.n_valid_samples is None:
                    # use all samples in valid set
                    self.data = valid_subset
                else:
                    # use a subset of valid set, useful to speedup evo search
                    valid_subset.indices = valid_subset.indices[
                                           :self.n_valid_samples]
                    self.data = valid_subset
                    logger.warning(f"Only use the front "
                                   f"{self.n_valid_samples} images as "
                                   f"VALID set.")

        else:
            test = datasets.CIFAR10(self.root,
                                      train=False, transform=transform)
            targets = torch.tensor(test.targets)
            idx, _ = torch.stack([targets == number for number in
                                  self.digits_of_interest]).max(dim=0)
            test.targets = targets[idx].numpy().tolist()
            test.data = test.data[idx]
            if self.n_test_samples is None:
                # use all samples as test set
                self.data = test
            else:
                # use a subset as test set
                test.targets = test.targets[:self.n_test_samples]
                test.data = test.data[:self.n_test_samples]
                self.data = test
                logger.warning(f"Only use the front {self.n_test_samples} "
                               f"images as TEST set.")

    def __getitem__(self, index: int):
        img = self.data[index][0]
        if self.binarize:
            img = 1. * (img > self.binarize_threshold) + \
                  -1. * (img <= self.binarize_threshold)

        digit = self.digits_of_interest.index(self.data[index][1])
        instance = {'image': img, 'digit': digit}
        return instance

    def __len__(self) -> int:
        return self.n_instance


class CIFAR10(Dataset):
    def __init__(self,
                 root: str,
                 train_valid_split_ratio: List[float],
                 center_crop=32,
                 resize=32,
                 resize_mode='bilinear',
                 binarize=False,
                 binarize_threshold=0.1307,
                 grayscale=False,
                 digits_of_interest=tuple(range(10)),
                 n_test_samples=None,
                 n_valid_samples=None,
                 fashion=False
                 ):
        self.root = root

        super().__init__({
            split: CIFAR10Dataset(
                root=root,
                split=split,
                train_valid_split_ratio=train_valid_split_ratio,
                center_crop=center_crop,
                resize=resize,
                resize_mode=resize_mode,
                binarize=binarize,
                binarize_threshold=binarize_threshold,
                grayscale=grayscale,
                digits_of_interest=digits_of_interest,
                n_test_samples=n_test_samples,
                n_valid_samples=n_valid_samples,
                fashion=fashion
            )
            for split in ['train', 'valid', 'test']
        })


if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    cifar10 = CIFAR10Dataset(root='../cifar10_data',
                         split='train',
                         train_valid_split_ratio=[0.9, 0.1],
                         center_crop=32,
                         resize=32,
                         resize_mode='bilinear',
                         binarize=False,
                         binarize_threshold=0.1307,
                         grayscale=True,
                         digits_of_interest=(3, 6),
                         n_test_samples=100,
                         n_valid_samples=1000,
                         fashion=True
                         )
    cifar10.__getitem__(20)
    print('finish')
