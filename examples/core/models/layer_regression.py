import torch.nn as nn
import torchquantum as tq
import torch
import numpy as np


class Measure(tq.QuantumModule):
    def __init__(self):
        super().__init__()

    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        x = tq.expval(q_device, list(range(q_device.n_wire)), [tq.PauliY()]
                      * q_device.n_wire)
        return x


class RxEncoder(tq.QuantumModule):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires
        self.rx_gates = tq.QuantumModuleList([tq.RX() for _ in range(
            self.n_wires)])

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        self.q_device.reset_states(bsz=x.shape[0])
        for k in range(self.n_wires):
            self.rx_gates[k](self.q_device, wires=k, params=x[:, k])


class LayerRegressionV(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wire=5)
        self.encoder = RxEncoder(n_wires=5)
        self.encoder.static_on(wires_per_block=5)
        self.measure = Measure()
        self.q_layer = tq.RandomLayer(n_ops=1000000,
                                      wires=list(range(5)))
        self.q_layer.static_on(wires_per_block=5)

    def forward(self, x):
        self.encoder(self.q_device, x)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)

        return x


class LayerRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_wire = 5
        self.q_device = tq.QuantumDevice(n_wire=5)
        self.encoder = RxEncoder(n_wires=5)
        self.encoder.static_on(wires_per_block=5)
        self.measure = Measure()
        self.qop = tq.Hadamard()
        self.q_layer = tq.RandomLayer(n_ops=3000,
                                      wires=list(range(5)))
        self.q_layer.static_on(wires_per_block=5)
        self.cnt = 0

    def forward(self, x):
        self.cnt += 1
        self.q_device.states = torch.eye(
            2 ** self.n_wire).to(self.q_device.state).view([32] + [2] *
                                                          self.n_wire)
        self.q_layer(self.q_device)
        # self.qop(self.q_device, wires=1)

        U_ = self.q_device.states.reshape(32, 32)

        check = False
        if check:
            U = self.q_device.states.view(32, 32).cpu().detach().numpy()
            ha = np.allclose(np.matmul(U, np.transpose(U.conj(), [1, 0])), np.identity(U.shape[0]), atol=1e-5)

        return U_.unsqueeze(0)
