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
from torch_geometric.nn import TransformerConv, global_mean_pool


class Simple_Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if self.args.use_global_features:
            self.gf_linear1 = torch.nn.Linear(6, 12)
            self.gf_linear2 = torch.nn.Linear(12, 12)
            self.linear1 = torch.nn.Linear(36, 128)
        else:
            self.linear1 = torch.nn.Linear(24, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, 1)
        self.mask = torch.full((24,), True)
        if not self.args.use_gate_type:
            self.mask[0:6] = False
        if not self.args.use_qubit_index:
            self.mask[6:16] = False
        if not self.args.use_T1T2:
            self.mask[16:20] = False
        if not self.args.use_gate_error:
            self.mask[20:23] = False
        if not self.args.use_gate_index:
            self.mask[23] = False
        xlenth = 0
        for i in self.mask:
            if i:
                xlenth += 1
        setattr(self, f"conv{0}", TransformerConv(xlenth, 24))
        for i in range(1, self.args.num_layers):
            setattr(self, f"conv{i}", TransformerConv(24, 24))

    def forward(self, data):
        x, edge_index, gf = data.x, data.edge_index, data.global_features
        x = x[:, self.mask]
        for i in range(self.args.num_layers):
            x = getattr(self, f"conv{i}")(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        if self.args.use_global_features:
            gf = self.gf_linear1(gf)
            gf = F.relu(gf)
            gf = self.gf_linear2(gf)
            gf = F.relu(gf)
            x = torch.cat([x, gf], dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x.squeeze()


class Global_Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear1 = torch.nn.Linear(6, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, 1)

    def forward(self, data):
        gf = data.global_features
        x = self.linear1(gf)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x.squeeze()


class Liu_Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear1 = torch.nn.Linear(116, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, 1)

    def forward(self, data):
        gf, liu = data.global_features, data.liu_features
        x = torch.cat([gf, liu], dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x.squeeze()
