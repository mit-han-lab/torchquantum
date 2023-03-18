#
# import torchquantum as tq
# import torchquantum.functional as tqf
#
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import random
#
# from torchquantum.datasets import MNIST
# from torch.optim.lr_scheduler import CosineAnnealingLR
#
#
# class QuanvolutionFilter(tq.QuantumModule):
#     def __init__(self):
#         super().__init__()
#         self.n_wires = 4
#         self.encoder = tq.GeneralEncoder(
#         [   {'input_idx': [0], 'func': 'ry', 'wires': [0]},
#             {'input_idx': [1], 'func': 'ry', 'wires': [1]},
#             {'input_idx': [2], 'func': 'ry', 'wires': [2]},
#             {'input_idx': [3], 'func': 'ry', 'wires': [3]},])
#
#         self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
#         self.measure = tq.MeasureAll(tq.PauliZ)
#
#     def forward(self, x, use_qiskit=False):
#         bsz = x.shape[0]
#         qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
#         size = 28
#         x = x.view(bsz, size, size)
#
#         data_list = []
#
#         for c in range(0, size, 2):
#             for r in range(0, size, 2):
#                 data = torch.transpose(torch.cat((x[:, c, r], x[:, c, r+1], x[:, c+1, r], x[:, c+1, r+1])).view(4, bsz), 0, 1)
#                 if use_qiskit:
#                     data = self.qiskit_processor.process_parameterized(
#                         qdev, self.encoder, self.q_layer, self.measure, data)
#                 else:
#                     self.encoder(qdev, data)
#                     self.q_layer(qdev)
#                     data = self.measure(qdev)
#
#                 data_list.append(data.view(bsz, 4))
#
#         result = torch.cat(data_list, dim=1).float()
#
#         return result
#
#
# class HybridModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.qf = QuanvolutionFilter()
#         self.linear = torch.nn.Linear(4*14*14, 10)
#
#     def forward(self, x, use_qiskit=False):
#         with torch.no_grad():
#           x = self.qf(x, use_qiskit)
#         x = self.linear(x)
#         return F.log_softmax(x, -1)
#
#
# class HybridModel_without_qf(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = torch.nn.Linear(28*28, 10)
#
#     def forward(self, x, use_qiskit=False):
#         x = x.view(-1, 28*28)
#         x = self.linear(x)
#         return F.log_softmax(x, -1)
#
#
# def train(dataflow, model, device, optimizer):
#     for feed_dict in dataflow['train']:
#         inputs = feed_dict['image'].to(device)
#         targets = feed_dict['digit'].to(device)
#
#         outputs = model(inputs)
#         loss = F.nll_loss(outputs, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f"loss: {loss.item()}", end='\r')
#
#
# def valid_test(dataflow, split, model, device, qiskit=False):
#     target_all = []
#     output_all = []
#     with torch.no_grad():
#         for feed_dict in dataflow[split]:
#             inputs = feed_dict['image'].to(device)
#             targets = feed_dict['digit'].to(device)
#
#             outputs = model(inputs, use_qiskit=qiskit)
#
#             target_all.append(targets)
#             output_all.append(outputs)
#         target_all = torch.cat(target_all, dim=0)
#         output_all = torch.cat(output_all, dim=0)
#
#     _, indices = output_all.topk(1, dim=1)
#     masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
#     size = target_all.shape[0]
#     corrects = masks.sum().item()
#     accuracy = corrects / size
#     loss = F.nll_loss(output_all, target_all).item()
#
#     print(f"{split} set accuracy: {accuracy}")
#     print(f"{split} set loss: {loss}")
#
#     return accuracy, loss
#
#
#
# def main():
#     train_model_without_qf = True
#     n_epochs = 15
#
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     dataset = MNIST(
#         root='./mnist_data',
#         train_valid_split_ratio=[0.9, 0.1],
#         n_test_samples=300,
#         n_train_samples=500,
#     )
#     dataflow = dict()
#
#     for split in dataset:
#         sampler = torch.utils.data.RandomSampler(dataset[split])
#         dataflow[split] = torch.utils.data.DataLoader(
#             dataset[split],
#             batch_size=10,
#             sampler=sampler,
#             num_workers=8,
#             pin_memory=True)
#
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     model = HybridModel().to(device)
#     model_without_qf = HybridModel_without_qf().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
#     scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
#
#     accu_list1 = []
#     loss_list1 = []
#     accu_list2 = []
#     loss_list2 = []
#     for epoch in range(1, n_epochs + 1):
#         # train
#         print(f"Epoch {epoch}:")
#         train(dataflow, model, device, optimizer)
#         print(optimizer.param_groups[0]['lr'])
#
#         # valid
#         accu, loss = valid_test(dataflow, 'test', model, device, )
#         accu_list1.append(accu)
#         loss_list1.append(loss)
#         scheduler.step()
#
#     if train_model_without_qf:
#         optimizer = optim.Adam(model_without_qf.parameters(), lr=5e-3, weight_decay=1e-4)
#         scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
#         for epoch in range(1, n_epochs + 1):
#             # train
#             print(f"Epoch {epoch}:")
#             train(dataflow, model_without_qf, device, optimizer)
#             print(optimizer.param_groups[0]['lr'])
#
#             # valid
#             accu, loss = valid_test(dataflow, 'test', model_without_qf, device)
#             accu_list2.append(accu)
#             loss_list2.append(loss)
#
#             scheduler.step()
#
#     # run on real QC
#     try:
#         from qiskit import IBMQ
#         from torchquantum.plugins import QiskitProcessor
#         # firstly perform simulate
#         print(f"\nTest with Qiskit Simulator")
#         processor_simulation = QiskitProcessor(use_real_qc=False)
#         model.qf.set_qiskit_processor(processor_simulation)
#         valid_test(dataflow, 'test', model, device, qiskit=True)
#         # then try to run on REAL QC
#         backend_name = 'ibmq_quito'
#         print(f"\nTest on Real Quantum Computer {backend_name}")
#         processor_real_qc = QiskitProcessor(use_real_qc=True, backend_name=backend_name)
#         model.qf.set_qiskit_processor(processor_real_qc)
#         valid_test(dataflow, 'test', model, device, qiskit=True)
#     except ImportError:
#         print("Please install qiskit, create an IBM Q Experience Account and "
#             "save the account token according to the instruction at "
#             "'https://github.com/Qiskit/qiskit-ibmq-provider', "
#             "then try again.")
#
#
# if __name__ == '__main__':
#     main()

import torch

from torchpack.datasets.dataset import Dataset
from torchvision import datasets, transforms
from typing import List
from torchpack.utils.logging import logger
from torchvision.transforms import InterpolationMode


__all__ = ["CIFAR10"]


resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


class CIFAR10Dataset:
    def __init__(
        self,
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
            tran = [
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize(
                    (0.2989 * 0.4914 + 0.587 * 0.4822 + 0.114 * 0.4465,),
                    (
                        (
                            (0.2989 * 0.2023) ** 2
                            + (0.587 * 0.1994) ** 2
                            + (0.114 * 0.2010) ** 2
                        )
                        ** 0.5,
                    ),
                ),
            ]
        else:
            tran = [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]

        if not self.center_crop == 32:
            tran.append(transforms.CenterCrop(self.center_crop))
        if not self.resize == 32:
            tran.append(transforms.Resize(self.resize, interpolation=self.resize_mode))
        transform = transforms.Compose(tran)

        if self.split == "train" or self.split == "valid":
            train_valid = datasets.CIFAR10(
                self.root, train=True, download=True, transform=transform
            )
            targets = torch.tensor(train_valid.targets)
            idx, _ = torch.stack(
                [targets == number for number in self.digits_of_interest]
            ).max(dim=0)
            # targets = targets[idx]
            train_valid.targets = targets[idx].numpy().tolist()
            train_valid.data = train_valid.data[idx]

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            split = [train_len, len(train_valid) - train_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )
            if self.split == "train":
                self.data = train_subset
            else:
                if self.n_valid_samples is None:
                    # use all samples in valid set
                    self.data = valid_subset
                else:
                    # use a subset of valid set, useful to speedup evo search
                    valid_subset.indices = valid_subset.indices[: self.n_valid_samples]
                    self.data = valid_subset
                    logger.warning(
                        f"Only use the front "
                        f"{self.n_valid_samples} images as "
                        f"VALID set."
                    )

        else:
            test = datasets.CIFAR10(self.root, train=False, transform=transform)
            targets = torch.tensor(test.targets)
            idx, _ = torch.stack(
                [targets == number for number in self.digits_of_interest]
            ).max(dim=0)
            test.targets = targets[idx].numpy().tolist()
            test.data = test.data[idx]
            if self.n_test_samples is None:
                # use all samples as test set
                self.data = test
            else:
                # use a subset as test set
                test.targets = test.targets[: self.n_test_samples]
                test.data = test.data[: self.n_test_samples]
                self.data = test
                logger.warning(
                    f"Only use the front {self.n_test_samples} " f"images as TEST set."
                )

    def __getitem__(self, index: int):
        img = self.data[index][0]
        if self.binarize:
            img = 1.0 * (img > self.binarize_threshold) + -1.0 * (
                img <= self.binarize_threshold
            )

        digit = self.digits_of_interest.index(self.data[index][1])
        instance = {"image": img, "digit": digit}
        return instance

    def __len__(self) -> int:
        return self.n_instance


class CIFAR10(Dataset):
    def __init__(
        self,
        root: str,
        train_valid_split_ratio: List[float],
        center_crop=32,
        resize=32,
        resize_mode="bilinear",
        binarize=False,
        binarize_threshold=0.1307,
        grayscale=False,
        digits_of_interest=tuple(range(10)),
        n_test_samples=None,
        n_valid_samples=None,
        fashion=False,
    ):
        self.root = root

        super().__init__(
            {
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
                    fashion=fashion,
                )
                for split in ["train", "valid", "test"]
            }
        )


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    cifar10 = CIFAR10Dataset(
        root="../cifar10_data",
        split="train",
        train_valid_split_ratio=[0.9, 0.1],
        center_crop=32,
        resize=32,
        resize_mode="bilinear",
        binarize=False,
        binarize_threshold=0.1307,
        grayscale=True,
        digits_of_interest=(3, 6),
        n_test_samples=100,
        n_valid_samples=1000,
        fashion=True,
    )
    cifar10.__getitem__(20)
    print("finish")
