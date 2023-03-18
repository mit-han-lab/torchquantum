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
