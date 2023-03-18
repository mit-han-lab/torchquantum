# import torch
# import torch.nn.functional as F
#
# import torchquantum as tq
#
# import random
# import numpy as np
#
# class QLayer(tq.QuantumModule):
#         def __init__(self):
#             super().__init__()
#             self.n_wires = 4
#             self.random_layer = tq.RandomLayer(n_ops=50,
#                                                seed=0,
#                                                wires=list(range(self.n_wires)))
#
#             # gates with trainable parameters
#             self.rx0 = tq.RX(has_params=True, trainable=True)
#             self.ry0 = tq.RY(has_params=True, trainable=True)
#             self.rz0 = tq.RZ(has_params=True, trainable=True)
#             self.crx0 = tq.CRX(has_params=True, trainable=True)
#
#         def forward(self, qdev: tq.QuantumDevice):
#             # self.random_layer(qdev)
#
#             # some trainable gates (instantiated ahead of time)
#             self.rx0(qdev, wires=0)
#             self.ry0(qdev, wires=1)
#             self.rz0(qdev, wires=3)
#             self.crx0(qdev, wires=[0, 2])
#
#             # add some more non-parameterized gates (add on-the-fly)
#             qdev.h(wires=3) # type: ignore
#             qdev.sx(wires=2) # type: ignore
#             qdev.cnot(wires=[3, 0]) # type: ignore
#             qdev.rx(wires=1, params=torch.tensor([0.1])) # type: ignore
#
# class QFCModel(tq.QuantumModule):
#     def __init__(self):
#         super().__init__()
#         self.n_wires = 4
#         self.encoder = tq.GeneralEncoder(
#             tq.encoder_op_list_name_dict['4x4_u3rx'])
#
#         self.q_layer = QLayer()
#         self.measure = tq.MeasureAll(tq.PauliZ)
#
#     def forward(self, x):
#         qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True)
#         bsz = x.shape[0]
#         x = F.avg_pool2d(x, 6).view(bsz, 16)
#
#         self.encoder(qdev, x)
#         self.q_layer(qdev)
#         x = self.measure(qdev)
#
#         x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
#         x = F.log_softmax(x, dim=1)
#
#         return x
#
#
# def save_load1():
#     """Assume the model construction code is available to the user."""
#     model = QFCModel()
#
#     x = torch.rand(2, 1, 28, 28)
#     y = model(x)
#     print(y)
#
#     torch.save(model.state_dict(), 'model_dict.pt')
#
#     model2 = QFCModel()
#     model2.load_state_dict(torch.load('model_dict.pt'))
#     y2 = model2(x)
#     print(y2)
#     assert torch.equal(y, y2)
#
#
# def save_load2():
#     """Assume the user doesn't want to create a new object.
#     In this case, the user should save and load the entire model.
#     WARNING: This will not work if the model class is not defined in the same file.
#     It will show the following error:
#         AttributeError: Can't get attribute 'QFCModel' on <module '__main__' from 'save_load.py'>
#     """
#     model = QFCModel()
#
#     x = torch.rand(2, 1, 28, 28)
#     y = model(x)
#     print(y)
#
#     torch.save(model, 'model_whole.pt')
#
#     model2 = torch.load('model_whole.pt')
#     y2 = model2(x)
#     print(y2)
#     assert torch.equal(y, y2)
#
# def save_load3():
#     """Assume the user cannot access to the model definition.
#     https://stackoverflow.com/questions/59287728/saving-pytorch-model-with-no-access-to-model-class-code
#     In this case, the user should save and load the entire model with torch.jit.script
#     """
#
#     model = QFCModel()
#
#     x = torch.rand(2, 1, 28, 28)
#     y = model(x)
#     print(y)
#
#     # the QFCModel class is not available to the user
#     torch.save(model, 'model_whole.pt')
#
#     # print(model.q_layer.rx0._parameters)
#
#     traced_cell = torch.jit.trace(model, (x))
#     torch.jit.save(traced_cell, "model_trace.pth")
#
#     loaded_trace = torch.jit.load("model_trace.pt")
#     y2 = loaded_trace(x)
#     print(y2)
#     assert torch.equal(y, y2)
#
#
# if __name__ == '__main__':
#     print(f"case 1: save and load the state_dict")
#     save_load1()
#     print(f"case 2: save and load the entire model")
#     save_load2()
#     print(f"case 3: save and load the entire model with torch.jit.script")
#     save_load3()

import torchquantum as tq
import torchquantum.functional as tqf

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from torchquantum.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR


from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.layers import U3CU3Layer0


class TrainableQuanvFilter(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        self.arch = {"n_wires": self.n_wires, "n_blocks": 5, "n_layers_per_block": 2}
        self.q_layer = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        x = F.avg_pool2d(x, 6).view(bsz, 4, 4)
        size = 4
        stride = 2
        x = x.view(bsz, size, size)

        data_list = []

        for c in range(0, size, stride):
            for r in range(0, size, stride):
                data = torch.transpose(
                    torch.cat(
                        (x[:, c, r], x[:, c, r + 1], x[:, c + 1, r], x[:, c + 1, r + 1])
                    ).view(4, bsz),
                    0,
                    1,
                )
                if use_qiskit:
                    data = self.qiskit_processor.process_parameterized(
                        qdev, self.encoder, self.q_layer, self.measure, data
                    )
                else:
                    self.encoder(qdev, data)
                    self.q_layer(qdev)
                    data = self.measure(qdev)

                data_list.append(data.view(bsz, 4))

        # transpose to (bsz, channel, 2x2)
        result = torch.transpose(
            torch.cat(data_list, dim=1).view(bsz, 4, 4), 1, 2
        ).float()

        return result


class QuantumClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict["4x4_ryzxy"])
        self.arch = {"n_wires": self.n_wires, "n_blocks": 8, "n_layers_per_block": 2}
        self.ansatz = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                qdev, self.encoder, self.q_layer, self.measure, x
            )  # type: ignore
        else:
            self.encoder(qdev, x)
            self.ansatz(qdev)
            x = self.measure(qdev)

        return x


class QFC(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict["4x4_ryzxy"])
        self.arch = {"n_wires": self.n_wires, "n_blocks": 4, "n_layers_per_block": 2}

        self.q_layer = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        data = x
        if use_qiskit:
            data = self.qiskit_processor.process_parameterized(
                qdev, self.encoder, self.q_layer, self.measure, data
            )
        else:
            self.encoder(qdev, data)
            self.q_layer(qdev)
            data = self.measure(qdev)
        return data


class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qf = TrainableQuanvFilter()
        self.linear = torch.nn.Linear(16, 4)

    def forward(self, x, use_qiskit=False):
        x = x.view(-1, 28, 28)
        x = self.qf(x)
        x = x.reshape(-1, 16)
        x = self.linear(x)
        return F.log_softmax(x, -1)


class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qf = TrainableQuanvFilter()
        self.qfc = QFC()

    def forward(self, x, use_qiskit=False):
        x = x.view(-1, 28, 28)
        x = self.qf(x)
        x = x.reshape(-1, 16)
        x = self.qfc(x)
        return F.log_softmax(x, -1)


class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qfc = QuantumClassifier()

    def forward(self, x, use_qiskit=False):
        x = self.qfc(x)
        return F.log_softmax(x, -1)


class Model4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 9)
        self.linear2 = torch.nn.Linear(9, 4)

    def forward(self, x, use_qiskit=False):
        x = x.view(-1, 28, 28)
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        x = self.linear1(x)
        x = self.linear2(x)
        return F.log_softmax(x, -1)


def train(dataflow, model, device, optimizer):
    for feed_dict in dataflow["train"]:
        inputs = feed_dict["image"].to(device)
        targets = feed_dict["digit"].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}", end="\r")


def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict["image"].to(device)
            targets = feed_dict["digit"].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")

    return accuracy, loss


def main():
    n_epochs = 1

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dataset = MNIST(
        root="./mnist_data",
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[0, 1, 2, 3],
        n_test_samples=300,
        n_train_samples=500,
    )

    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=10,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
        )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    accus = []

    model_list = [
        Model1().to(device),
        Model2().to(device),
        Model3().to(device),
        Model4().to(device),
    ]

    for i, model in enumerate(model_list):
        print(f"training model {i}...")
        optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
        for epoch in range(1, n_epochs + 1):
            # train
            print(f"Epoch {epoch}:")
            train(dataflow, model, device, optimizer)
            print(optimizer.param_groups[0]["lr"])
            # valid
            accu, loss = valid_test(dataflow, "test", model, device)
            scheduler.step()
        accus.append(accu)

    for i, accu in enumerate(accus):
        print("accuracy of model{0}: {1}".format(i + 1, accu))


if __name__ == "__main__":
    main()
