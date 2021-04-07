import torchquantum as tq
import torch.nn.functional as F
from torchpack.utils.logging import logger


class Q4DigitFCModel0(tq.QuantumModule):
    """rx ry rz crx cry crz layers"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = arch['n_wires']
            self.rx_layers = tq.QuantumModuleList()
            self.ry_layers = tq.QuantumModuleList()
            self.rz_layers = tq.QuantumModuleList()
            self.crx_layers = tq.QuantumModuleList()
            self.cry_layers = tq.QuantumModuleList()
            self.crz_layers = tq.QuantumModuleList()

            for k in range(arch['n_blocks']):
                self.rx_layers.append(
                    tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.ry_layers.append(
                    tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.rz_layers.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.crx_layers.append(
                    tq.Op2QAllLayer(op=tq.CRX, n_wires=self.n_wires,
                                    has_params=True, trainable=True,
                                    circular=True))
                self.cry_layers.append(
                    tq.Op2QAllLayer(op=tq.CRY, n_wires=self.n_wires,
                                    has_params=True, trainable=True,
                                    circular=True))
                self.crz_layers.append(
                    tq.Op2QAllLayer(op=tq.CRZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True,
                                    circular=True))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device

            for k in range(self.arch['n_blocks']):
                self.rx_layers[k](self.q_device)
                self.ry_layers[k](self.q_device)
                self.rz_layers[k](self.q_device)
                self.crx_layers[k](self.q_device)
                self.cry_layers[k](self.q_device)
                self.crz_layers[k](self.q_device)

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

    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = x.squeeze()
        x = F.log_softmax(x, dim=1)

        return x


class Q4DigitFCModel1(Q4DigitFCModel0):
    """u3 and cu3 layers, one layer of u3 and one layer of cu3 in one block"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = arch['n_wires']
            self.u3_layers = tq.QuantumModuleList()
            self.cu3_layers = tq.QuantumModuleList()

            for k in range(arch['n_blocks']):
                self.u3_layers.append(
                    tq.Op1QAllLayer(op=tq.U3, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.cu3_layers.append(
                    tq.Op2QAllLayer(op=tq.CU3, n_wires=self.n_wires,
                                    has_params=True, trainable=True,
                                    circular=True))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.arch['n_blocks']):
                self.u3_layers[k](self.q_device)
                self.cu3_layers[k](self.q_device)


class Q4DigitFCModel2(Q4DigitFCModel0):
    """ry and cz layers, one layer of ry and one layer of cz in one block"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = arch['n_wires']
            self.ry_layers = tq.QuantumModuleList()
            self.cz_layers = tq.QuantumModuleList()

            for k in range(arch['n_blocks']):
                self.ry_layers.append(
                    tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.cz_layers.append(
                    tq.Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires,
                                    circular=True))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.arch['n_blocks']):
                self.ry_layers[k](self.q_device)
                self.cz_layers[k](self.q_device)


class Q4DigitFCRandomModel0(Q4DigitFCModel0):
    """model with random gates"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = arch['n_wires']
            op_type_name = arch['op_type_name']
            op_types = [tq.op_name_dict[name] for name in op_type_name]

            self.random_layer = tq.RandomLayer(
                n_ops=arch['n_random_ops'],
                n_params=arch['n_random_params'],
                wires=list(range(self.n_wires)),
                op_types=op_types)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.random_layer(q_device)


model_dict = {
    'q4digit_fc0': Q4DigitFCModel0,
    'q4digit_fc1': Q4DigitFCModel1,
    'q4digit_fc2': Q4DigitFCModel2,
    'q4digit_fc_rand0': Q4DigitFCRandomModel0,
}
