from torchpack.datasets.dataset import Dataset
import torch
import numpy as np

class Classification2Dataset(torch.utils.data.Dataset):
    def __init__(self, num=11):
        self.data = []
        self.target = []
        sum0 = 0
        sum1 = 0
        for x in np.linspace(0, 1, num=num):
            for y in np.linspace(0, 1, num=num):
                self.data.append(torch.tensor([x, y]))
                if (x**2 + y**2 <= 0.55**2 or (x-1)**2 + (y-1)**2 <= 0.55**2):
                    self.target.append(1)
                    sum1 = sum1 + 1
                else:
                    self.target.append(0)
                    sum0 = sum0 + 1
            print(self.target[-num:])

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'target': self.target[idx]}

    def __len__(self):
        return len(self.target)

class Simple2Class(Dataset):
    def __init__(self):
        train_dataset = Classification2Dataset()
        valid_dataset = Classification2Dataset(num=10)
        datasets = {'train': train_dataset, 'valid': valid_dataset, 'test': valid_dataset}
        super().__init__(datasets)
