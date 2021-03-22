import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F


class SuperQFCModel0(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.encoder = tq.MultiPhaseEncoder([tqf.rx] * 4 + [tqf.ry] * 4 +
                                            [tqf.rz] * 4 + [tqf.rx] * 4)

        self.super_layers_all = tq.QuantumModuleList()

        for k in range(4):
            self.super_layers_all.append(
                tq.Super1QLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.super_layers_all.append(
                tq.Super1QLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.super_layers_all.append(
                tq.Super1QLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.super_layers_all.append(
                tq.Super2QLayer(op=tq.CNOT, n_wires=self.n_wires))

    def set_sample_arch(self, sample_arch):
        for k, layer_config in enumerate(sample_arch):
            self.super_layers_all[k].set_sample_arch(layer_config)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(self.q_device, x)

        for k in range(len(self.super_layers_all)):
            self.super_layers_all[k](self.q_device)

        x = self.measure(self.q_device).reshape(bsz, 2, 2)
        x = x.sum(-1).squeeze()

        x = F.log_softmax(x, dim=1)

        return x

    @property
    def arch_space(self):
        space = []
        for layer in self.super_layers_all:
            space.append(layer.arch_space)
        return space


class SuperQFCModel1(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.encoder = tq.MultiPhaseEncoder([tqf.rx] * 4 + [tqf.ry] * 4 +
                                            [tqf.rz] * 4 + [tqf.rx] * 4)

        self.super_layers_all = tq.QuantumModuleList()

        for k in range(4):
            self.super_layers_all.append(
                tq.Super1QShareFrontLayer(op=tq.RX, n_wires=self.n_wires,
                                          n_front_share_wires=2,
                                          has_params=True, trainable=True))
            self.super_layers_all.append(
                tq.Super1QShareFrontLayer(op=tq.RY, n_wires=self.n_wires,
                                          n_front_share_wires=2,
                                          has_params=True, trainable=True))
            self.super_layers_all.append(
                tq.Super1QShareFrontLayer(op=tq.RZ, n_wires=self.n_wires,
                                          n_front_share_wires=2,
                                          has_params=True, trainable=True))
            # self.super_cnot_layers.append(
            #     tq.Super2QLayer(op=tq.CNOT, n_wires=self.n_wires))

    def set_sample_arch(self, sample_arch):
        for k, layer_config in enumerate(sample_arch):
            self.super_layers_all[k].set_sample_arch(layer_config)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(self.q_device, x)

        for k in range(len(self.super_layers_all)):
            self.super_layers_all[k](self.q_device)

            if k % 3 == 1:
                tqf.cnot(self.q_device, wires=[0, 1])
                tqf.cnot(self.q_device, wires=[2, 3])
                tqf.cnot(self.q_device, wires=[k % 4, (k + 1) % 4])

        x = self.measure(self.q_device).reshape(bsz, 2, 2)
        x = x.sum(-1).squeeze()

        x = F.log_softmax(x, dim=1)

        return x

    @property
    def arch_space(self):
        space = []
        for layer in self.super_layers_all:
            space.append(layer.arch_space)
        return space


class SuperQFCModel2(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.encoder = tq.MultiPhaseEncoder([tqf.rx] * 4 + [tqf.ry] * 4 +
                                            [tqf.rz] * 4 + [tqf.rx] * 4)

        self.super_layers_all = tq.QuantumModuleList()
        self.normal_layers_all = tq.QuantumModuleList()

        for k in range(2):
            self.normal_layers_all.append(
                tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True)
            )
            self.normal_layers_all.append(
                tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True)
            )
            self.normal_layers_all.append(
                tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True)
            )

        for k in range(2):
            self.super_layers_all.append(
                tq.Super1QAllButOneLayer(op=tq.RX, n_wires=self.n_wires,
                                         has_params=True, trainable=True))
            self.super_layers_all.append(
                tq.Super1QAllButOneLayer(op=tq.RY, n_wires=self.n_wires,
                                         has_params=True, trainable=True))
            self.super_layers_all.append(
                tq.Super1QAllButOneLayer(op=tq.RZ, n_wires=self.n_wires,
                                         has_params=True, trainable=True))
            self.super_layers_all.append(
                tq.Super1QSingleWireLayer(op=tq.RX, n_wires=self.n_wires,
                                          has_params=True, trainable=True))
            self.super_layers_all.append(
                tq.Super1QSingleWireLayer(op=tq.RY, n_wires=self.n_wires,
                                          has_params=True, trainable=True))
            self.super_layers_all.append(
                tq.Super1QSingleWireLayer(op=tq.RZ, n_wires=self.n_wires,
                                          has_params=True, trainable=True))

    def set_sample_arch(self, sample_arch):
        for k, layer_config in enumerate(sample_arch):
            self.super_layers_all[k].set_sample_arch(layer_config)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(self.q_device, x)

        for k in range(len(self.super_layers_all)):
            self.super_layers_all[k](self.q_device)

            if k % 3 == 2:
                tqf.cnot(self.q_device, wires=[0, 1])
                tqf.cnot(self.q_device, wires=[2, 3])
                tqf.cnot(self.q_device, wires=[k % 4, (k + 1) % 4])
            if k % 6 == 5:
                self.normal_layers_all[0 + k // 6 * 3](self.q_device)
                self.normal_layers_all[1 + k // 6 * 3](self.q_device)
                self.normal_layers_all[2 + k // 6 * 3](self.q_device)

        x = self.measure(self.q_device).reshape(bsz, 2, 2)
        x = x.sum(-1).squeeze()

        x = F.log_softmax(x, dim=1)

        return x

    @property
    def arch_space(self):
        space = []
        for layer in self.super_layers_all:
            space.append(layer.arch_space)
        return space


model_dict = {
    'super_qfc0': SuperQFCModel0,
    'super_qfc1': SuperQFCModel1,
    'super_qfc2': SuperQFCModel2,
}
