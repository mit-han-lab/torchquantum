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
import torch.optim as optim
import numpy as np

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.layer.layers import SethLayer0

from torchquantum.dataset import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR


class QFCModel(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])

        self.arch = {"n_wires": self.n_wires, "n_blocks": 2, "n_layers_per_block": 2}
        self.q_layer = SethLayer0(self.arch)

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        q_device = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz)
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                q_device, self.encoder, self.q_layer, self.measure, x
            )
        else:
            self.encoder(q_device, x)
            self.q_layer(q_device)
            x = self.measure(q_device)

        x = x.reshape(bsz, 4)

        return x


def shift_and_run(model, inputs, use_qiskit=False):
    param_list = []
    for param in model.parameters():
        param_list.append(param)
    grad_list = []
    for param in param_list:
        param.copy_(param + np.pi * 0.5)
        out1 = model(inputs, use_qiskit)
        param.copy_(param - np.pi)
        out2 = model(inputs, use_qiskit)
        param.copy_(param + np.pi * 0.5)
        grad = 0.5 * (out1 - out2)
        grad_list.append(grad)
    return model(inputs, use_qiskit), grad_list


def main():
    with torch.no_grad():
        outputs, grad_list = shift_and_run(
            QFCModel(), torch.randn(2, 1, 28, 28), use_qiskit=False
        )
    print(outputs, grad_list)


if __name__ == "__main__":
    main()
