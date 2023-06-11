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
import torchquantum as tq

# import pdb
# pdb.set_trace()

encoder = tq.GeneralEncoder(
    [
        {"input_idx": [0], "func": "ry", "wires": [0]},
        {"input_idx": [1], "func": "ry", "wires": [1]},
        {"input_idx": [2], "func": "ry", "wires": [2]},
        {"input_idx": [3], "func": "ry", "wires": [3]},
        {"input_idx": [4], "func": "ry", "wires": [4]},
        {"input_idx": [5], "func": "ry", "wires": [5]},
        {"input_idx": [6], "func": "ry", "wires": [6]},
        {"input_idx": [7], "func": "ry", "wires": [7]},
        {"input_idx": [8], "func": "ry", "wires": [0]},
        {"input_idx": [9], "func": "ry", "wires": [1]},
        {"input_idx": [10], "func": "ry", "wires": [2]},
        {"input_idx": [11], "func": "ry", "wires": [3]},
        {"input_idx": [12], "func": "ry", "wires": [4]},
        {"input_idx": [13], "func": "ry", "wires": [5]},
        {"input_idx": [14], "func": "ry", "wires": [6]},
        {"input_idx": [15], "func": "ry", "wires": [7]},
    ]
)

bsz = 10

qdev = tq.QuantumDevice(n_wires=8, bsz=bsz)

x = torch.rand(bsz, 16)
encoder(qdev, x)

print(qdev)
