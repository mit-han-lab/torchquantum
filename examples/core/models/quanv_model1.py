import torchquantum as tq
import torch
import torch.nn.functional as F
import numpy as np

__all__ = ['QuanvModel1']


class Quanv0(tq.QuantumModule):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(
            self.n_wires)))

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
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires
        # self.h_gates = [tq.Hadamard() for _ in range(self.n_wires)]
        self.rx_gates = tq.QuantumModuleList([tq.RX() for _ in range(
            self.n_wires)])

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        self.q_device.reset_states(bsz=x.shape[0])
        # for k in range(self.n_wires):
        #     self.h_gates[k](self.q_device, wires=k)
        for k in range(self.n_wires):
            self.rx_gates[k](self.q_device, wires=k, params=x[:, k])


class QuanvModel1(tq.QuantumModule):
    """
    Convolution with quantum filter
    """
    def __init__(self):
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wire=4)
        self.measure = Measure()
        self.wires_per_block = 4
        self.n_quanv = 3

        self.encoder0 = RxEncoder(n_wires=4)
        # self.encoder0.static_on(wires_per_block=self.wires_per_block)
        self.quanv0_all = tq.QuantumModuleList()
        for k in range(self.n_quanv):
            self.quanv0_all.append(Quanv0(n_wires=4))
            # self.quanv0[k].static_on(wires_per_block=self.wires_per_block)

        self.quanv1_all = tq.QuantumModuleList()
        # self.encoder1.static_on(wires_per_block=self.wires_per_block)
        for k in range(10):
            self.quanv1_all.append(Quanv0(n_wires=4))
            # self.quanv1[k].static_on(wires_per_block=self.wires_per_block)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6)

        x = F.unfold(x, kernel_size=2, stride=1)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, x.shape[-1])
        x = F.tanh(x) * np.pi

        for k in range(self.n_quanv):
            self.encoder0(self.q_device, x)
            self.quanv0_all[k](self.q_device)
            x = self.measure(self.q_device)
            x = x * np.pi

        # x = x.view(bsz, 3, 3, 4).permute(0, 3, 1, 2)

        # for k in range(3):
        #     self.encoder0(self.q_device, x)
        #     self.quanv0[k](self.q_device)
        #     x = self.measure(self.q_device)
        #     quanv0_results.append(x.sum(-1).view(bsz, 13, 13))
        # x = torch.stack(quanv0_results, dim=1)

        # x = F.unfold(x, kernel_size=2, stride=2)
        # x = x.permute(0, 2, 1)
        # x = x.reshape(-1, x.shape[-1])

        quanv1_results = []
        for k in range(10):
            self.encoder0(self.q_device, x)
            self.quanv1_all[k](self.q_device)
            x = self.measure(self.q_device)
            quanv1_results.append(x.sum(-1).view(bsz, 3, 3))
        x = torch.stack(quanv1_results, dim=1)

        x = F.avg_pool2d(x, kernel_size=3)
        x = F.log_softmax(x, dim=1)
        x = x.squeeze()

        return x
