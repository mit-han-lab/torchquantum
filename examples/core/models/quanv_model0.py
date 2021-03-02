import torchquantum as tq
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QuanvModel0']


class Quanv0(tq.QuantumModule):
    def __init__(self, n_gates):
        super().__init__()
        self.n_gates = n_gates
        self.random_layer = tq.RandomLayer(n_ops=200, wires=list(range(
            self.n_gates)))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        self.random_layer(self.q_device)


class Measure(tq.QuantumModule):
    def __init__(self):
        super().__init__()

    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        x = tq.expval(q_device, list(range(q_device.n_wire)), [tq.PauliZ()]
                      * q_device.n_wire)
        return x


class RxEncoder(tq.QuantumModule):
    def __init__(self, n_gates):
        super().__init__()
        self.n_gates = n_gates
        self.h_gates = [tq.Hadamard() for _ in range(self.n_gates)]
        self.rx_gates = [tq.RX() for _ in range(self.n_gates)]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        self.q_device.reset_states(bsz=x.shape[0])
        for k in range(self.n_gates):
            self.h_gates[k](self.q_device, wires=k)
        for k in range(self.n_gates):
            self.rx_gates[k](self.q_device, wires=k, params=x[:, k])


class QuanvModel0(tq.QuantumModule):
    """
    Convolution with quantum filter
    """
    def __init__(self):
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wire=9)
        self.encoder0 = RxEncoder(n_gates=9)
        self.quanv0 = Quanv0(n_gates=9)
        self.measure0 = Measure()

        # self.fc1 = nn.Linear(13 * 13, 10)
        # self.fc1 = nn.Linear(13 * 13, 10)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.unfold(x, kernel_size=3)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, x.shape[-1])
        self.encoder0(self.q_device, x)
        self.quanv0(self.q_device)
        x = self.measure0(self.q_device)
        x = x.sum(-1).view(bsz, 26, 26)








        # out0 = torch.empty([x.shape[0], 9, 26, 26]).to(x)

        # need to change from conv 2d dim to batch dim to increase speed!

        # for j in range(0, 7):
        #     for k in range(0, 7):
        #         q_layer0_in[:, j, k, :] = \
        #             torch.stack([
        #                 x[:, 0, j, k],
        #                 x[:, 0, j, k + 1],
        #                 x[:, 0, j, k + 2],
        #                 x[:, 0, j + 1, k],
        #                 x[:, 0, j + 1, k + 1],
        #                 x[:, 0, j + 1, k + 2],
        #                 x[:, 0, j + 1, k],
        #                 x[:, 0, j + 1, k + 1],
        #                 x[:, 0, j + 1, k + 2]
        #             ], dim=-1)
        #         # out0[:, :, j, k] = q_results
        # q_layer0_in = q_layer0_in.view(bsz * 7 * 7, 9)
        # q_layer0_out = self.q_layer0(q_layer0_in)
        # q_layer0_out = q_layer0_out.view(bsz, 7, 7, 9).permute(0, 3, 1, 2)
        #
        # # x = self.conv1(q_layer0_out)
        # x = torch.sum(q_layer0_out, dim=-3)

        # out1 = torch.empty([x.shape[0], 9, 24, 24]).to(x)
        # for j in range(0, 24):
        #     for k in range(0, 24):
        #         q_results = self.q_layer1(
        #             torch.stack([
        #                 x[:, :, j, k],
        #                 x[:, :, j, k + 1],
        #                 x[:, :, j, k + 2],
        #                 x[:, :, j + 1, k],
        #                 x[:, :, j + 1, k + 1],
        #                 x[:, :, j + 1, k + 2],
        #                 x[:, :, j + 1, k],
        #                 x[:, :, j + 1, k + 1],
        #                 x[:, :, j + 1, k + 2]
        #             ], dim=-1)
        #         )
        #         out1[:, :, j, k] = q_results

        # x = self.conv1(out0)
        # x = F.max_pool2d(x, 2, padding=1)
        # x = torch.flatten(x, 1)
        # x = F.relu(x)
        # # x = self.dropout2(x)
        # x = self.fc0(x)
        # x = x[:, :10]

        # output = F.log_softmax(x, dim=1)

        return x
