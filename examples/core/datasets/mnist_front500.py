import torch

from torchpack.datasets.dataset import Dataset
from torchvision import datasets, transforms
from typing import List
from torchpack.utils.logging import logger
from torchvision.transforms import InterpolationMode


__all__ = ['MNIST_FRONT500']


resize_modes = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'nearest': InterpolationMode.NEAREST,
}


class MNISTDataset:
    def __init__(self,
                 root: str,
                 split: str,
                 train_valid_split_ratio: List[float],
                 center_crop,
                 resize,
                 resize_mode,
                 binarize,
                 binarize_threshold,
                 digits_of_interest,
                 n_test_samples,
                 n_valid_samples,
                 fashion,
                 front_size,
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
        self.digits_of_interest = digits_of_interest
        self.n_test_samples = n_test_samples
        self.n_valid_samples = n_valid_samples
        self.fashion = fashion
        self.front_size = front_size

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        tran = [transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))]
        if not self.center_crop == 28:
            tran.append(transforms.CenterCrop(self.center_crop))
        if not self.resize == 28:
            tran.append(transforms.Resize(self.resize,
                                          interpolation=self.resize_mode))
        transform = transforms.Compose(tran)

        if self.split == 'train' or self.split == 'valid':
            if self.fashion:
                train_valid = datasets.FashionMNIST(
                    self.root, train=True, download=True, transform=transform)
            else:
                train_valid = datasets.MNIST(
                    self.root, train=True, download=True, transform=transform)
            # idx, _ = torch.stack([train_valid.targets == number for number in
            #                       self.digits_of_interest]).max(dim=0)
            # train_valid.targets = train_valid.targets[idx][:500]
            # train_valid.data = train_valid.data[idx][:500, :, :]
            num_samples_one_class = self.front_size // len(self.digits_of_interest)
            targets_list = []
            data_list = []
            num_mix = 1
            fold = 2
            for number in self.digits_of_interest:
                idx = (train_valid.targets == number)
                targets_list.append(train_valid.targets[idx][:num_samples_one_class])
                # data_list.append(train_valid.data[idx][:num_samples_one_class, :, :].float().to(dtype=torch.uint8))
                data_list.append(torch.mean(train_valid.data[idx][:fold * num_samples_one_class, :, :].view(fold, num_samples_one_class, 28, 28).float(), dim=0).to(dtype=torch.uint8))
            train_valid.targets = torch.cat(targets_list, dim=0)
            train_valid.data = torch.cat(data_list, dim=0)

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
            if self.fashion:
                test = datasets.FashionMNIST(self.root,
                                             train=False, transform=transform)
            else:
                test = datasets.MNIST(self.root,
                                      train=False, transform=transform)
            idx, _ = torch.stack([test.targets == number for number in
                                  self.digits_of_interest]).max(dim=0)
            test.targets = test.targets[idx]
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


class MNIST_FRONT500(Dataset):
    def __init__(self,
                 root: str,
                 train_valid_split_ratio: List[float],
                 center_crop=28,
                 resize=28,
                 resize_mode='bilinear',
                 binarize=False,
                 binarize_threshold=0.1307,
                 digits_of_interest=tuple(range(10)),
                 n_test_samples=None,
                 n_valid_samples=None,
                 fashion=False,
                 front_size=500,
                 ):
        self.root = root

        super().__init__({
            split: MNISTDataset(
                root=root,
                split=split,
                train_valid_split_ratio=train_valid_split_ratio,
                center_crop=center_crop,
                resize=resize,
                resize_mode=resize_mode,
                binarize=binarize,
                binarize_threshold=binarize_threshold,
                digits_of_interest=digits_of_interest,
                n_test_samples=n_test_samples,
                n_valid_samples=n_valid_samples,
                fashion=fashion,
                front_size=front_size,
            )
            for split in ['train', 'valid', 'test']
        })


if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    mnist = MNISTDataset(root='../fashion_mnist_data',
                         split='train',
                         train_valid_split_ratio=[0.9, 0.1],
                         center_crop=28,
                         resize=28,
                         resize_mode='bilinear',
                         binarize=False,
                         binarize_threshold=0.1307,
                         digits_of_interest=(3, 6),
                         n_test_samples=100,
                         n_valid_samples=1000,
                         fashion=True
                         )
    mnist.__getitem__(20)
    print('finish')
