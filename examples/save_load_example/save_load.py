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

import torch
import torch.nn.functional as F

import torchquantum as tq

import random
import numpy as np


class QLayer(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(
            n_ops=50, seed=0, wires=list(range(self.n_wires))
        )

        # gates with trainable parameters
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        # self.random_layer(qdev)

        # some trainable gates (instantiated ahead of time)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])

        # add some more non-parameterized gates (add on-the-fly)
        qdev.h(wires=3)  # type: ignore
        qdev.sx(wires=2)  # type: ignore
        qdev.cnot(wires=[3, 0])  # type: ignore
        qdev.rx(wires=1, params=torch.tensor([0.1]))  # type: ignore


class QFCModel(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_u3rx"])

        self.q_layer = QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True
        )
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        self.encoder(qdev, x)
        self.q_layer(qdev)
        x = self.measure(qdev)

        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x


def save_load1():
    """Assume the model construction code is available to the user."""
    model = QFCModel()

    x = torch.rand(2, 1, 28, 28)
    y = model(x)
    print(y)

    torch.save(model.state_dict(), "model_dict.pt")

    model2 = QFCModel()
    model2.load_state_dict(torch.load("model_dict.pt"))
    y2 = model2(x)
    print(y2)
    assert torch.equal(y, y2)


def save_load2():
    """Assume the user doesn't want to create a new object.
    In this case, the user should save and load the entire model.
    WARNING: This will not work if the model class is not defined in the same file.
    It will show the following error:
        AttributeError: Can't get attribute 'QFCModel' on <module '__main__' from 'save_load.py'>
    """
    model = QFCModel()

    x = torch.rand(2, 1, 28, 28)
    y = model(x)
    print(y)

    torch.save(model, "model_whole.pt")

    model2 = torch.load("model_whole.pt", weights_only=False)
    y2 = model2(x)
    print(y2)
    assert torch.equal(y, y2)


def save_load3():
    """Assume the user cannot access to the model definition.
    https://stackoverflow.com/questions/59287728/saving-pytorch-model-with-no-access-to-model-class-code
    In this case, the user should save and load the entire model with torch.jit.script
    """

    model = QFCModel()

    x = torch.rand(2, 1, 28, 28)
    y = model(x)
    print(y)

    # the QFCModel class is not available to the user
    torch.save(model, "model_whole.pt")

    # print(model.q_layer.rx0._parameters)

    traced_cell = torch.jit.trace(model, (x))
    torch.jit.save(traced_cell, "model_trace.pt")

    loaded_trace = torch.jit.load("model_trace.pt")
    y2 = loaded_trace(x)
    print(y2)
    assert torch.equal(y, y2)


if __name__ == "__main__":
    print(f"case 1: save and load the state_dict")
    save_load1()
    print(f"case 2: save and load the entire model")
    save_load2()
    print(f"case 3: save and load the entire model with torch.jit.script")
    save_load3()
