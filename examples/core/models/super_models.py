import torch
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

from torchpack.utils.logging import logger
from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.super_layers import super_layer_name_dict


class SuperQFCModel0(tq.QuantumModule):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            encoder_op_list_name_dict[arch['encoder_op_list_name']]
        )
        self.q_layer = super_layer_name_dict[arch['q_layer_name']](arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.sample_arch = None

    def set_sample_arch(self, sample_arch):
        self.sample_arch = sample_arch
        self.q_layer.set_sample_arch(sample_arch)

    def count_sample_params(self):
        return self.q_layer.count_sample_params()

    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]

        if getattr(self.arch, 'down_sample_kernel_size', None) is not None:
            x = F.avg_pool2d(x, self.arch['down_sample_kernel_size'])

        x = x.view(bsz, -1)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        if getattr(self.arch, 'output_len', None) is not None:
            x = x.reshape(bsz, -1, self.arch.output_len).sum(-1)

        if x.dim() > 2:
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


class SuperVQEModel0(tq.QuantumModule):
    def __init__(self, arch, hamil_info):
        super().__init__()
        self.arch = arch
        self.hamil_info = hamil_info
        self.n_wires = arch['n_wires']
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = super_layer_name_dict[arch['q_layer_name']](arch)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])

        self.sample_arch = None

    def set_sample_arch(self, sample_arch):
        self.sample_arch = sample_arch
        self.q_layer.set_sample_arch(sample_arch)

    def count_sample_params(self):
        return self.q_layer.count_sample_params()

    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                           self.hamil_info['hamil_list']],
                                          device=x.device)

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1]
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

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


class SuperQFCModel1(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires):
            super().__init__()
            self.n_wires = n_wires
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

        def set_sample_arch(self, sample_arch):
            for k, layer_arch in enumerate(sample_arch):
                self.super_layers_all[k].set_sample_arch(layer_arch)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x):
            self.q_device = q_device
            self.encoder(self.q_device, x)

            for k in range(len(self.super_layers_all)):
                self.super_layers_all[k](self.q_device)

                if k % 3 == 1:
                    tqf.cnot(self.q_device, wires=[0, 1],
                             static=self.static_mode, parent_graph=self.graph)
                    tqf.cnot(self.q_device, wires=[2, 3],
                             static=self.static_mode, parent_graph=self.graph)
                    tqf.cnot(self.q_device, wires=[k % 4, (k + 1) % 4],
                             static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.q_layer = self.QLayer(n_wires=self.n_wires)

    def set_sample_arch(self, sample_arch):
        self.q_layer.set_sample_arch(sample_arch)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process(self.q_device, self.q_layer, x)
        else:
            self.q_layer(self.q_device, x)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x

    @property
    def arch_space(self):
        space = []
        for layer in self.q_layer.super_layers_all:
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
        for k, layer_arch in enumerate(sample_arch):
            self.super_layers_all[k].set_sample_arch(layer_arch)

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


class SuperQFCModel3(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, arch):
            super().__init__()
            self.n_wires = n_wires
            self.arch = arch

            self.super_layers_all = tq.QuantumModuleList()
            self.n_blocks = arch['n_blocks']
            self.n_front_share_wires = arch['n_front_share_wires']
            self.n_front_share_ops = arch['n_front_share_ops']
            self.n_front_share_blocks = arch['n_front_share_blocks']

            for k in range(self.n_blocks):
                self.super_layers_all.append(
                    tq.Super1QShareFrontLayer(
                        op=tq.RX, n_wires=self.n_wires,
                        n_front_share_wires=self.n_front_share_wires,
                        has_params=True, trainable=True))
                self.super_layers_all.append(
                    tq.Super1QShareFrontLayer(
                        op=tq.RY, n_wires=self.n_wires,
                        n_front_share_wires=self.n_front_share_wires,
                        has_params=True, trainable=True))
                self.super_layers_all.append(
                    tq.Super1QShareFrontLayer(
                        op=tq.RZ, n_wires=self.n_wires,
                        n_front_share_wires=self.n_front_share_wires,
                        has_params=True, trainable=True))
                self.super_layers_all.append(
                    tq.Super2QAllShareFrontLayer(
                        op=tq.CRX,
                        n_wires=self.n_wires,
                        n_front_share_ops=self.n_front_share_ops,
                        has_params=True,
                        trainable=True,
                        jump=1,
                        circular=True))
                self.super_layers_all.append(
                    tq.Super2QAllShareFrontLayer(
                        op=tq.CRY,
                        n_wires=self.n_wires,
                        n_front_share_ops=self.n_front_share_ops,
                        has_params=True,
                        trainable=True,
                        jump=1,
                        circular=True))
                self.super_layers_all.append(
                    tq.Super2QAllShareFrontLayer(
                        op=tq.CRZ,
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
                if k < self.sample_n_blocks * 6:
                    self.super_layers_all[k](self.q_device)

    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = 4
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
        self.q_layer = self.QLayer(n_wires=self.n_wires, arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def set_sample_arch(self, sample_arch):
        self.q_layer.set_sample_arch(sample_arch)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
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


class SuperQFCModel4(tq.QuantumModule):
    """different from model3: no share front"""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, arch):
            super().__init__()
            self.n_wires = n_wires
            self.arch = arch

            self.super_layers_all = tq.QuantumModuleList()
            self.n_blocks = arch['n_blocks']
            self.n_layers_per_block = arch['n_layers_per_block']
            self.n_front_share_blocks = arch['n_front_share_blocks']

            for k in range(self.n_blocks):
                self.super_layers_all.append(
                    tq.Super1QLayer(
                        op=tq.RX, n_wires=self.n_wires,
                        has_params=True, trainable=True))
                self.super_layers_all.append(
                    tq.Super1QLayer(
                        op=tq.RY, n_wires=self.n_wires,
                        has_params=True, trainable=True))
                self.super_layers_all.append(
                    tq.Super1QLayer(
                        op=tq.RZ, n_wires=self.n_wires,
                        has_params=True, trainable=True))
                self.super_layers_all.append(
                    tq.Super2QAllLayer(
                        op=tq.CRX,
                        n_wires=self.n_wires,
                        has_params=True,
                        trainable=True,
                        jump=1,
                        circular=True))
                self.super_layers_all.append(
                    tq.Super2QAllLayer(
                        op=tq.CRY,
                        n_wires=self.n_wires,
                        has_params=True,
                        trainable=True,
                        jump=1,
                        circular=True))
                self.super_layers_all.append(
                    tq.Super2QAllLayer(
                        op=tq.CRZ,
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
                    self.super_layers_all[k](self.q_device)

    def __init__(self, arch=None):
        super().__init__()
        self.n_wires = 4
        self.arch = arch
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
        self.q_layer = self.QLayer(n_wires=self.n_wires, arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def set_sample_arch(self, sample_arch):
        self.q_layer.set_sample_arch(sample_arch)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
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


model_dict = {
    'super_qfc0': SuperQFCModel0,
    'super_qfc1': SuperQFCModel1,
    'super_qfc2': SuperQFCModel2,
    'super_qfc3': SuperQFCModel3,
    'super_qfc4': SuperQFCModel4,
    'supervqe_0': SuperVQEModel0,
}
