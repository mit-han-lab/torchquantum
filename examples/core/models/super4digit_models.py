import torchquantum as tq
import torch.nn.functional as F


class Super4DigitShareFrontQFCModel1(tq.QuantumModule):
    """u3 cu3 blocks"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = arch['n_wires']

            self.n_front_share_wires = arch['n_front_share_wires']
            self.n_front_share_ops = arch['n_front_share_ops']

            self.n_blocks = arch['n_blocks']
            self.n_layers_per_block = arch['n_layers_per_block']
            self.n_front_share_blocks = arch['n_front_share_blocks']

            self.super_layers_all = tq.QuantumModuleList()

            for k in range(arch['n_blocks']):
                self.super_layers_all.append(
                    tq.Super1QShareFrontLayer(
                        op=tq.U3,
                        n_wires=self.n_wires,
                        n_front_share_wires=self.n_front_share_wires,
                        has_params=True,
                        trainable=True))
                self.super_layers_all.append(
                    tq.Super2QAllShareFrontLayer(
                        op=tq.CU3,
                        n_wires=self.n_wires,
                        n_front_share_ops=self.n_front_share_ops,
                        has_params=True,
                        trainable=True,
                        jump=1,
                        circular=True))
            self.sample_n_blocks = None

        def set_sample_arch(self, sample_arch):
            for k, layer_arch in enumerate(sample_arch[:-1]):
                self.super_layers_all[k].set_sample_arch(layer_arch)
            self.sample_n_blocks = sample_arch[-1]

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(len(self.super_layers_all)):
                if k < self.sample_n_blocks * self.n_layers_per_block:
                    self.super_layers_all[k](q_device)

    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
            {'input_idx': [2], 'func': 'ry', 'wires': [2]},
            {'input_idx': [3], 'func': 'ry', 'wires': [3]},
            {'input_idx': [4], 'func': 'rz', 'wires': [0]},
            {'input_idx': [5], 'func': 'rz', 'wires': [1]},
            {'input_idx': [6], 'func': 'rz', 'wires': [2]},
            {'input_idx': [7], 'func': 'rz', 'wires': [3]},
            {'input_idx': [8], 'func': 'rx', 'wires': [0]},
            {'input_idx': [9], 'func': 'rx', 'wires': [1]},
            {'input_idx': [10], 'func': 'rx', 'wires': [2]},
            {'input_idx': [11], 'func': 'rx', 'wires': [3]},
            {'input_idx': [12], 'func': 'ry', 'wires': [0]},
            {'input_idx': [13], 'func': 'ry', 'wires': [1]},
            {'input_idx': [14], 'func': 'ry', 'wires': [2]},
            {'input_idx': [15], 'func': 'ry', 'wires': [3]}
        ])
        self.q_layer = self.QLayer(arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def set_sample_arch(self, sample_arch):
        self.q_layer.set_sample_arch(sample_arch)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(self.q_device, x)

        self.q_layer(self.q_device)
        x = self.measure(self.q_device)

        x = x.squeeze()
        x = F.log_softmax(x, dim=1)

        return x

    def forward_qiskit(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        x = self.qiskit_processor.process_parameterized(
            self.q_device, self.encoder, self.q_layer, x)

        x = x.squeeze()
        x = F.log_softmax(x, dim=1)

        return x

    @property
    def arch_space(self):
        space = []
        for layer in self.q_layer.super_layers_all:
            space.append(layer.arch_space)
        # for the number of sampled blocks
        space.append(list(range(self.q_layer.n_front_share_blocks,
                                self.q_layer.n_blocks + 1)))
        return space


class Super4DigitArbitraryQFCModel1(Super4DigitShareFrontQFCModel1):
    """u3 cu3 blocks arbitrary n gates"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = arch['n_wires']

            self.n_blocks = arch['n_blocks']
            self.n_layers_per_block = arch['n_layers_per_block']
            self.n_front_share_blocks = arch['n_front_share_blocks']

            self.super_layers_all = tq.QuantumModuleList()

            for k in range(arch['n_blocks']):
                self.super_layers_all.append(
                    tq.Super1QLayer(
                        op=tq.U3,
                        n_wires=self.n_wires,
                        has_params=True,
                        trainable=True))
                self.super_layers_all.append(
                    tq.Super2QAllLayer(
                        op=tq.CU3,
                        n_wires=self.n_wires,
                        has_params=True,
                        trainable=True,
                        jump=1,
                        circular=True))
            self.sample_n_blocks = None

        def set_sample_arch(self, sample_arch):
            for k, layer_arch in enumerate(sample_arch[:-1]):
                self.super_layers_all[k].set_sample_arch(layer_arch)
            self.sample_n_blocks = sample_arch[-1]

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(len(self.super_layers_all)):
                if k < self.sample_n_blocks * self.n_layers_per_block:
                    self.super_layers_all[k](q_device)


model_dict = {
    'super4digit_sharefront_fc1': Super4DigitShareFrontQFCModel1,
    'super4digit_arbitrary_fc1': Super4DigitArbitraryQFCModel1,
}
