"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from copy import deepcopy

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torchpack.utils.config import configs

from core.datasets import builder
from core.datasets.drawer import draw_curve, draw_scatter


class trainer:
    def __init__(self, model, device, criterion, optimizer, scheduler, loaders):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loaders = loaders
        self.best = 1e10
        self.best_params = None
        self.training_data = {}

    def train(self):
        self.model.train()
        self.training_data["train_loss"] = []
        self.training_data["val_error"] = []

        for epoch in range(configs.num_epochs):
            loss_sum = 0
            for batch in self.loaders["train"]:
                self.optimizer.zero_grad()
                out = self.model(batch)
                loss = self.criterion(out, batch.y.to(self.device))
                loss.backward()
                self.optimizer.step()
                loss_sum += (
                    loss.item() * len(batch.y) / len(self.loaders["train"].dataset)
                )
            self.scheduler.step()
            print(
                f"[{epoch + 1} / {configs.num_epochs}],sqrtloss={loss_sum**0.5} \r",
                end="",
            )
            self.training_data["train_loss"].append(loss_sum**0.5)
            if epoch % 5 == 0:
                val_error = self.valid()
                self.save_best(val_error)
                self.training_data["val_error"].append(val_error)
        torch.save(self.best_params, f"exp/{configs.exp_name}/model.pth")
        print("\n")

    def save_best(self, loss):
        if loss < self.best:
            self.best = loss
            self.best_params = deepcopy(self.model.state_dict())

    def valid(self):
        self.model.eval()
        pred_error = 0
        for batch in self.loaders["valid"]:
            out = self.model(batch)
            pred_error += ((out - batch.y.to(self.device)) ** 2).sum().item()
        pred_error = (pred_error / len(self.loaders["valid"].dataset)) ** 0.5
        print(
            f"\t\t\t\t\t\t val_error:{pred_error} \r",
            end="",
        )
        return pred_error

    def saveall(self):
        mydict = {}
        mydict["train_loss"] = self.training_data["train_loss"]
        mydict["val_error"] = self.training_data["val_error"]
        mydict["test_pred"] = self.training_data["test_pred"]
        mydict["test_y"] = self.training_data["test_y"]
        mydict["test_error"] = self.test_error
        mydict["best"] = self.best

        torch.save(mydict, f"exp/{configs.exp_name}/all.pth")

    def loadall(self):
        mydict = torch.load(f"exp/{configs.exp_name}/all.pth")
        self.training_data["train_loss"] = mydict["train_loss"]
        self.training_data["val_error"] = mydict["val_error"]
        self.training_data["test_pred"] = mydict["test_pred"]
        self.training_data["test_y"] = mydict["test_y"]
        self.test_error = mydict["test_error"]
        self.best = mydict["best"]
        print(f"test_error:{self.test_error}")
        print(f"best_val_error:{self.best}")

    def test(self):
        self.training_data["test_pred"] = np.array([])
        self.training_data["test_y"] = np.array([])
        self.test_error = 0
        print(len(self.loaders["test"].dataset))
        if len(self.loaders["test"].dataset) > 1:
            self.model.load_state_dict(torch.load(f"exp/{configs.exp_name}/model.pth"))
            self.model.eval()
            test_error = 0
            for batch in self.loaders["test"]:
                out = self.model(batch)
                self.training_data["test_pred"] = np.concatenate(
                    (self.training_data["test_pred"], out.cpu().detach().numpy())
                )
                self.training_data["test_y"] = np.concatenate(
                    (self.training_data["test_y"], batch.y.cpu().detach().numpy())
                )
                test_error += ((out - batch.y.to(self.device)) ** 2).sum().item()
            test_error = (test_error / len(self.loaders["test"].dataset)) ** 0.5
            self.test_error = test_error

    def testwith(self, dataname, device):
        self.training_data["test_pred_with" + dataname] = np.array([])
        self.training_data["test_y_with" + dataname] = np.array([])
        testdataset = builder.make_dataset_from(dataname).get_data(device, "test")
        testdataset = DataLoader(testdataset, batch_size=configs.batch_size)

        self.model.load_state_dict(torch.load(f"exp/{configs.exp_name}/model.pth"))
        self.model.eval()
        test_error = 0
        for batch in testdataset:
            out = self.model(batch)
            self.training_data["test_pred_with" + dataname] = np.concatenate(
                (
                    self.training_data["test_pred_with" + dataname],
                    out.cpu().detach().numpy(),
                )
            )
            self.training_data["test_y_with" + dataname] = np.concatenate(
                (
                    self.training_data["test_y_with" + dataname],
                    batch.y.cpu().detach().numpy(),
                )
            )
            test_error += ((out - batch.y.to(self.device)) ** 2).sum().item()
        test_error = (test_error / len(testdataset.dataset)) ** 0.5
        self.training_data["test_error_with" + dataname] = test_error
        print(f"test_error:{self.test_error}")
        # draw_scatter(
        #     self.training_data["test_y_with"],
        #     self.training_data["test_pred_with"],
        #     name="test.png",
        # )

    def save_training_data(self):
        torch.save(self.training_data, f"exp/{configs.exp_name}/training_data.pth")

    def scatter(self):
        draw_scatter(
            self.training_data["test_y"],
            self.training_data["test_pred"],
        )

    def curve(self):
        draw_curve(
            np.arange(len(self.training_data["train_loss"])),
            self.training_data["train_loss"],
        )
