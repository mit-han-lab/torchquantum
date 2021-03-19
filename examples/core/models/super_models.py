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

        self.super_all = tq.QuantumModuleDict()
        self.super_rx_layers = tq.QuantumModuleList()
        self.super_ry_layers = tq.QuantumModuleList()
        self.super_rz_layers = tq.QuantumModuleList()
        self.super_cnot_layers = tq.QuantumModuleList()

        for k in range(4):
            self.super_rx_layers.append(
                tq.Super1QLayer(op=tq.RX, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.super_ry_layers.append(
                tq.Super1QLayer(op=tq.RY, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.super_rz_layers.append(
                tq.Super1QLayer(op=tq.RZ, n_wires=self.n_wires,
                                has_params=True, trainable=True))
            self.super_cnot_layers.append(
                tq.Super2QLayer(op=tq.CNOT, n_wires=self.n_wires))

        self.super_all['rx'] = self.super_rx_layers
        self.super_all['ry'] = self.super_ry_layers
        self.super_all['rz'] = self.super_rz_layers
        self.super_all['cnot'] = self.super_cnot_layers

    def set_sample_config(self, sample_config):
        for name, layer_configs in sample_config.items():
            for k, layer_config in enumerate(layer_configs):
                self.super_all[name][k].set_sample_config(layer_config)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(self.q_device, x)

        for k in range(4):
            self.super_rx_layers[k](self.q_device)
            self.super_ry_layers[k](self.q_device)
            self.super_rz_layers[k](self.q_device)
            self.super_cnot_layers[k](self.q_device)

        x = self.measure(self.q_device).reshape(bsz, 2, 2)
        x = x.sum(-1).squeeze()

        x = F.log_softmax(x, dim=1)

        return x

    @property
    def config_space(self):
        space = {}
        for name, layers in self.super_all.items():
            space[name] = []
            for layer in layers:
                space[name].append(layer.config_space)

        return space


class SuperQFCModel1(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.encoder = tq.MultiPhaseEncoder([tqf.rx] * 4 + [tqf.ry] * 4 +
                                            [tqf.rz] * 4 + [tqf.rx] * 4)

        self.super_all = tq.QuantumModuleDict()
        self.super_rx_layers = tq.QuantumModuleList()
        self.super_ry_layers = tq.QuantumModuleList()
        self.super_rz_layers = tq.QuantumModuleList()
        self.super_cnot_layers = tq.QuantumModuleList()

        for k in range(4):
            self.super_rx_layers.append(
                tq.Super1QShareFrontLayer(op=tq.RX, n_wires=self.n_wires,
                                          n_front_share_wires=3,
                                          has_params=True, trainable=True))
            self.super_ry_layers.append(
                tq.Super1QShareFrontLayer(op=tq.RY, n_wires=self.n_wires,
                                          n_front_share_wires=3,
                                          has_params=True, trainable=True))
            self.super_rz_layers.append(
                tq.Super1QShareFrontLayer(op=tq.RZ, n_wires=self.n_wires,
                                          n_front_share_wires=3,
                                          has_params=True, trainable=True))
            # self.super_cnot_layers.append(
            #     tq.Super2QLayer(op=tq.CNOT, n_wires=self.n_wires))

        self.super_all['rx'] = self.super_rx_layers
        self.super_all['ry'] = self.super_ry_layers
        self.super_all['rz'] = self.super_rz_layers
        # self.super_all['cnot'] = self.super_cnot_layers

    def set_sample_config(self, sample_config):
        for name, layer_configs in sample_config.items():
            for k, layer_config in enumerate(layer_configs):
                self.super_all[name][k].set_sample_config(layer_config)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(self.q_device, x)

        for k in range(4):
            self.super_rx_layers[k](self.q_device)
            self.super_ry_layers[k](self.q_device)
            self.super_rz_layers[k](self.q_device)
            tqf.cnot(self.q_device, wires=[0, 1])
            tqf.cnot(self.q_device, wires=[2, 3])
            tqf.cnot(self.q_device, wires=[k, (k + 1) % 4])

            # self.super_cnot_layers[k](self.q_device)

        x = self.measure(self.q_device).reshape(bsz, 2, 2)
        x = x.sum(-1).squeeze()

        x = F.log_softmax(x, dim=1)

        return x

    @property
    def config_space(self):
        space = {}
        for name, layers in self.super_all.items():
            space[name] = []
            for layer in layers:
                space[name].append(layer.config_space)

        return space


model_dict = {
    'super_qfc0': SuperQFCModel0,
    'super_qfc1': SuperQFCModel1,
}
