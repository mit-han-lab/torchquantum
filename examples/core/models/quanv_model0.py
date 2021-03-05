import torchquantum as tq
import torch
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
        # self.h_gates = [tq.Hadamard() for _ in range(self.n_gates)]
        self.rx_gates = tq.QuantumModuleList([tq.RX() for _ in range(
            self.n_gates)])

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        self.q_device.reset_states(bsz=x.shape[0])
        # for k in range(self.n_gates):
        #     self.h_gates[k](self.q_device, wires=k)
        for k in range(self.n_gates):
            self.rx_gates[k](self.q_device, wires=k, params=x[:, k])


class QuanvModel0(tq.QuantumModule):
    """
    Convolution with quantum filter
    """
    def __init__(self):
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wire=9)
        self.q_device1 = tq.QuantumDevice(n_wire=12)
        self.measure = Measure()
        self.wires_per_block = 3

        self.encoder0 = RxEncoder(n_gates=9)
        self.encoder0.static_on(wires_per_block=self.wires_per_block)
        self.quanv0 = tq.QuantumModuleList()
        for k in range(3):
            self.quanv0.append(Quanv0(n_gates=9))
            self.quanv0[k].static_on(wires_per_block=self.wires_per_block)

        self.quanv1 = tq.QuantumModuleList()
        self.encoder1 = RxEncoder(n_gates=12)
        self.encoder1.static_on(wires_per_block=self.wires_per_block)
        for k in range(10):
            self.quanv1.append(Quanv0(n_gates=12))
            self.quanv1[k].static_on(wires_per_block=self.wires_per_block)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.unfold(x, kernel_size=3, stride=2)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, x.shape[-1])

        quanv0_results = []
        for k in range(3):
            self.encoder0(self.q_device, x)
            self.quanv0[k](self.q_device)
            x = self.measure(self.q_device)
            quanv0_results.append(x.sum(-1).view(bsz, 13, 13))
        x = torch.stack(quanv0_results, dim=1)

        x = F.unfold(x, kernel_size=2, stride=2)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, x.shape[-1])

        quanv1_results = []
        for k in range(10):
            self.encoder1(self.q_device1, x)
            self.quanv1[k](self.q_device1)
            x = self.measure(self.q_device1)
            quanv1_results.append(x.sum(-1).view(bsz, 6, 6))
        x = torch.stack(quanv1_results, dim=1)

        x = F.avg_pool2d(x, kernel_size=6)
        x = F.log_softmax(x, dim=1)
        x = x.squeeze()

        return x
