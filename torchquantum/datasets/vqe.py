from torchpack.datasets.dataset import Dataset


__all__ = ['VQE']


class VQEDataset:
    def __init__(self,
                 split,
                 steps_per_epoch
                 ):
        self.split = split
        self.steps_per_epoch = steps_per_epoch

    def __getitem__(self, index: int):
        instance = {'input': -1, 'target': -1}

        return instance

    def __len__(self) -> int:
        if self.split == 'train':
            return self.steps_per_epoch
        else:
            return 1


class VQE(Dataset):
    def __init__(self, steps_per_epoch):
        super().__init__({
            split: VQEDataset(
                split=split,
                steps_per_epoch=steps_per_epoch
            )
            for split in ['train', 'valid', 'test']
        })
