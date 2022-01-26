from torchpack.datasets.dataset import Dataset
import torch
import numpy as np

class Classification3Dataset(torch.utils.data.Dataset):
    def __init__(self, num=11):
        self.data = []
        self.target = []
        delta = 0.4
        sum0 = 0
        sum1 = 0
        sum2 = 0
        for x in np.linspace(0, 1, num=num):
            for y in np.linspace(0, 1, num=num):
                for z in np.linspace(0, 1, num=num):
                    if (x > y + delta and x > z + delta):
                        self.data.append(torch.tensor([x, y, z]))
                        self.target.append(2)
                        sum0 = sum0 + 1
                    elif (y > x + delta and y > z + delta):
                        self.data.append(torch.tensor([x, y, z]))
                        self.target.append(0)
                        sum1 = sum1 + 1
                    elif (z > x + delta and z > y + delta):
                        self.data.append(torch.tensor([x, y, z]))
                        self.target.append(1)
                        sum2 = sum2 + 1
            #     self.data.append(torch.tensor([x, y]))
            #     if (y <= 0.3):
            #         self.target.append(0)
            #         sum0 = sum0 + 1
            #     elif (1 - y <= 0.3):
            #         self.target.append(2)
            #         sum2 = sum2 + 1
            #     else:
            #         self.target.append(1)
            #         sum1 = sum1 + 1
            # print(self.target[-num:])
        print(sum0)
        print(sum1)
        print(sum2)

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'target': self.target[idx]}

    def __len__(self):
        return len(self.target)

class Simple3Class(Dataset):
    def __init__(self):
        train_dataset = Classification3Dataset(num=5)
        valid_dataset = Classification3Dataset(num=5)
        datasets = {'train': train_dataset, 'valid': valid_dataset, 'test': valid_dataset}
        super().__init__(datasets)
