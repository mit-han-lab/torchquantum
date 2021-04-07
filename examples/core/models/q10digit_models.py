import torchquantum as tq
import torch.nn.functional as F
from torchpack.utils.logging import logger


class Q10DigitFCModel0(tq.QuantumModule):
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
            {'input_idx': [4], 'func': 'ry', 'wires': [4]},
            {'input_idx': [5], 'func': 'ry', 'wires': [5]},
            {'input_idx': [6], 'func': 'ry', 'wires': [6]},
            {'input_idx': [7], 'func': 'ry', 'wires': [7]},
            {'input_idx': [8], 'func': 'ry', 'wires': [8]},
            {'input_idx': [9], 'func': 'ry', 'wires': [9]},
            {'input_idx': [10], 'func': 'rz', 'wires': [0]},
            {'input_idx': [11], 'func': 'rz', 'wires': [1]},
            {'input_idx': [12], 'func': 'rz', 'wires': [2]},
            {'input_idx': [13], 'func': 'rz', 'wires': [3]},
            {'input_idx': [14], 'func': 'rz', 'wires': [4]},
            {'input_idx': [15], 'func': 'rz', 'wires': [5]},
            {'input_idx': [16], 'func': 'rz', 'wires': [6]},
            {'input_idx': [17], 'func': 'rz', 'wires': [7]},
            {'input_idx': [18], 'func': 'rz', 'wires': [8]},
            {'input_idx': [19], 'func': 'rz', 'wires': [9]},
            {'input_idx': [20], 'func': 'rx', 'wires': [0]},
            {'input_idx': [21], 'func': 'rx', 'wires': [1]},
            {'input_idx': [22], 'func': 'rx', 'wires': [2]},
            {'input_idx': [23], 'func': 'rx', 'wires': [3]},
            {'input_idx': [24], 'func': 'rx', 'wires': [4]},
            {'input_idx': [25], 'func': 'rx', 'wires': [5]},
            {'input_idx': [26], 'func': 'rx', 'wires': [6]},
            {'input_idx': [27], 'func': 'rx', 'wires': [7]},
            {'input_idx': [28], 'func': 'rx', 'wires': [8]},
            {'input_idx': [29], 'func': 'rx', 'wires': [9]},
            {'input_idx': [30], 'func': 'ry', 'wires': [0]},
            {'input_idx': [31], 'func': 'ry', 'wires': [1]},
            {'input_idx': [32], 'func': 'ry', 'wires': [2]},
            {'input_idx': [33], 'func': 'ry', 'wires': [3]},
            {'input_idx': [34], 'func': 'ry', 'wires': [4]},
            {'input_idx': [35], 'func': 'ry', 'wires': [5]},
        ])
        self.q_layer = self.QLayer(arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 4).view(bsz, 36)

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


model_dict = {
    'q10digit_fc0': Q10DigitFCModel0,
}
