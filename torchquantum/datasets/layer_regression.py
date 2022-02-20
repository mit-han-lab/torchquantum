from torchpack.datasets.dataset import Dataset
import numpy as np
import torch

__all__ = ['LayerRegression']


class LayerRegressionDatasetV:
    def __init__(self):
        self.data = None
        self.n_instance = 10000
        self.input = np.random.rand(5).astype(np.float) * 2 * np.pi
        self.output = np.random.rand(5).astype(np.float) * 2 - 1

    def __getitem__(self, index: int):
        instance = {'input': self.input,
                    'output': self.output}
        return instance

    def __len__(self) -> int:
        return self.n_instance


class LayerRegressionV(Dataset):
    def __init__(self,):
        super().__init__({
            split: LayerRegressionDatasetV()
            for split in ['train', 'valid', 'test']
        })


class LayerRegressionDataset:
    def __init__(self):
        self.data = None
        self.n_instance = 10000
        mat = torch.randn((2 ** 5, 2 ** 5), dtype=torch.complex64)
        mat = mat.svd()[0].data

        self.output = mat

    def __getitem__(self, index: int):
        instance = {'input': self.output,
                    'output': self.output}
        return instance

    def __len__(self) -> int:
        return self.n_instance


class LayerRegression(Dataset):
    def __init__(self,):
        super().__init__({
            split: LayerRegressionDataset()
            for split in ['train', 'valid', 'test']
        })