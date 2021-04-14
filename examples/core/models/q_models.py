import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn.functional as F
import numpy as np

from torchquantum.encoding import encoder_op_list_name_dict
from torchpack.utils.logging import logger
from examples.core.tools.generate_ansatz_observables import (
    molecule_name_dict, generate_uccsd)


class Quanv0(tq.QuantumModule):
    def __init__(self, n_wires, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=200, wires=list(range(
            self.n_wires)))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        self.random_layer(self.q_device)


class QuanvModel0(tq.QuantumModule):
    """
    Convolution with quantum filter
    """
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.q_device = tq.QuantumDevice(n_wires=9)
        self.q_device1 = tq.QuantumDevice(n_wires=12)
        self.measure = tq.MeasureAll(obs=tq.PauliZ)
        self.wires_per_block = 5

        self.encoder0 = tq.PhaseEncoder(func=tqf.rx)
        self.encoder0.static_on(wires_per_block=self.wires_per_block)
        self.quanv0 = tq.QuantumModuleList()
        for k in range(3):
            self.quanv0.append(Quanv0(n_wires=9))
            self.quanv0[k].static_on(wires_per_block=self.wires_per_block)

        self.quanv1 = tq.QuantumModuleList()
        self.encoder1 = tq.PhaseEncoder(func=tqf.rx)
        self.encoder1.static_on(wires_per_block=self.wires_per_block)
        for k in range(10):
            self.quanv1.append(Quanv0(n_wires=12))
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


class QuanvModel1(tq.QuantumModule):
    """
    Convolution with quantum filter
    """
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.q_device = tq.QuantumDevice(n_wires=4)
        self.measure = tq.MeasureAll(obs=tq.PauliZ)
        self.wires_per_block = 4
        self.n_quanv = 3

        self.encoder0 = tq.PhaseEncoder(func=tqf.rx)
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


class QFCModel0(tq.QuantumModule):
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.q_device = tq.QuantumDevice(n_wires=4)
        self.encoder = tq.StateEncoder()
        self.trainable_u = tq.TrainableUnitary(has_params=True,
                                               trainable=True,
                                               n_wires=4)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        self.encoder(self.q_device, x)
        self.trainable_u(self.q_device, wires=[0, 1, 2, 3])

        x = self.q_device.states.view(bsz, 16)[:, :self.arch[
            'output_len']].abs()

        x = F.log_softmax(x, dim=1)

        return x


class QFCModel1(tq.QuantumModule):
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.q_device = tq.QuantumDevice(n_wires=4)
        self.encoder = tq.StateEncoder()
        self.trainable_u = tq.TrainableUnitary(has_params=True,
                                               trainable=True,
                                               n_wires=4)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        self.encoder(self.q_device, x)
        self.trainable_u(self.q_device, wires=[0, 1, 2, 3])

        x = self.measure(self.q_device).view(bsz, 4)[:, :self.arch[
            'output_len']]

        x = F.log_softmax(x, dim=1)

        return x


class QFCModel2(tq.QuantumModule):
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = 16
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.PhaseEncoder(tqf.rx)
        self.trainable_u = tq.TrainableUnitary(has_params=True,
                                               trainable=True,
                                               n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        self.encoder(self.q_device, x)
        self.trainable_u(self.q_device, wires=list(range(self.n_wires)))
        x = self.q_device.states.view(bsz, 16)[:, :self.arch[
            'output_len']].abs()

        x = F.log_softmax(x, dim=1)

        return x


class QFCModel3(tq.QuantumModule):
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.q_device = tq.QuantumDevice(n_wires=10)
        self.encoder = tq.StateEncoder()
        self.trainable_u = tq.TrainableUnitary(has_params=True,
                                               trainable=True,
                                               n_wires=10)
        self.trainable_u1 = tq.TrainableUnitary(has_params=True,
                                                trainable=True,
                                                n_wires=10)
        # if configs.regularization.unitary_loss_lambda_trainable:
        #     unitary_loss_lambda = nn.Parameter(
        #         torch.ones(1) * configs.regularization.unitary_loss_lambda)
        #     self.register_parameter('unitary_loss_lambda',
        #                             unitary_loss_lambda)

    def forward(self, x):
        bsz = x.shape[0]
        x = x.view(bsz, 784)

        self.encoder(self.q_device, x)
        self.trainable_u(self.q_device, wires=list(range(10)))
        self.trainable_u1(self.q_device, wires=list(range(10)))

        x = self.q_device.states.view(bsz, 1024)[:, :10].abs()

        x = F.log_softmax(x, dim=1)

        return x


class QuanvModel2(tq.QuantumModule):
    """
    Convolution with quantum filter
    """
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.q_device = tq.QuantumDevice(n_wires=4)
        self.measure = tq.MeasureAll(obs=tq.PauliZ)
        self.encoder = tq.PhaseEncoder(func=tqf.rx)

        self.quanv0 = tq.TrainableUnitary(has_params=True,
                                          trainable=True,
                                          n_wires=4)

        self.quanv1 = tq.TrainableUnitary(has_params=True,
                                          trainable=True,
                                          n_wires=4)

        self.quanv2 = tq.QuantumModuleList()
        for k in range(2):
            self.quanv2.append(
                tq.TrainableUnitary(has_params=True,
                                    trainable=True,
                                    n_wires=4)
            )

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6)

        x = F.unfold(x, kernel_size=2, stride=1)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, x.shape[-1])
        x = F.tanh(x) * np.pi

        self.encoder(self.q_device, x)
        self.quanv0(self.q_device, wires=[0, 1, 2, 3])
        x = self.measure(self.q_device)
        x = x * np.pi

        self.encoder(self.q_device, x)
        self.quanv1(self.q_device, wires=[0, 1, 2, 3])
        x = self.measure(self.q_device)
        x = x * np.pi

        x = x.view(bsz, 3, 3, 4)
        x = x.permute(0, 3, 1, 2)

        quanv2_results = []
        for k in range(2):
            tmp = x[:, k, :, :].unsqueeze(1)
            tmp = F.unfold(tmp, kernel_size=2, stride=1)  # bsz, 4, 4
            tmp = tmp.permute(0, 2, 1)
            tmp = tmp.reshape(-1, tmp.shape[-1])
            self.encoder(self.q_device, tmp)
            self.quanv2[k](self.q_device, wires=[0, 1, 2, 3])
            tmp = self.measure(self.q_device)
            quanv2_results.append(tmp.sum(-1).view(bsz, 2, 2))
        x = torch.stack(quanv2_results, dim=1)

        x = F.avg_pool2d(x, kernel_size=2
                         )[:, :self.arch['output_len']]
        x = F.log_softmax(x, dim=1)
        x = x.squeeze()

        return x


class QuanvModel3(tq.QuantumModule):
    """
    Convolution with quantum filter
    """
    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.q_device = tq.QuantumDevice(n_wires=4)
        self.n_wires = 4
        self.measure = tq.MeasureAll(obs=tq.PauliZ)
        self.encoder = tq.PhaseEncoder(func=tqf.rx)

        self.quanv0 = tq.RandomLayer(n_ops=200, wires=list(range(
            self.n_wires)))
        self.quanv0.static_on(wires_per_block=2)

        self.quanv1 = tq.RandomLayer(n_ops=200, wires=list(range(
            self.n_wires)))

        self.quanv2 = tq.QuantumModuleList()
        for k in range(2):
            self.quanv2.append(
                tq.RandomLayer(n_ops=200, wires=list(range(
                    self.n_wires)))
            )

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6)

        x = F.unfold(x, kernel_size=2, stride=1)
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, x.shape[-1])
        x = F.tanh(x) * np.pi

        self.encoder(self.q_device, x)
        self.quanv0(self.q_device)
        x = self.measure(self.q_device)
        x = x * np.pi

        self.encoder(self.q_device, x)
        self.quanv1(self.q_device)
        x = self.measure(self.q_device)
        x = x * np.pi

        x = x.view(bsz, 3, 3, 4)
        x = x.permute(0, 3, 1, 2)

        quanv2_results = []
        for k in range(2):
            tmp = x[:, k, :, :].unsqueeze(1)
            tmp = F.unfold(tmp, kernel_size=2, stride=1)  # bsz, 4, 4
            tmp = tmp.permute(0, 2, 1)
            tmp = tmp.reshape(-1, tmp.shape[-1])
            self.encoder(self.q_device, tmp)
            self.quanv2[k](self.q_device)
            tmp = self.measure(self.q_device)
            quanv2_results.append(tmp.sum(-1).view(bsz, 2, 2))
        x = torch.stack(quanv2_results, dim=1)

        x = F.avg_pool2d(x, kernel_size=2
                         )[:, :self.arch['output_len']]
        x = F.log_softmax(x, dim=1)
        x = x.squeeze()

        return x


# Qubitization according to the Quantum Singular Value Transformation paper
class QSVT0(tq.QuantumModule):
    def __init__(self,
                 n_wires=8,
                 n_xcnot_wires=8,
                 depth=16,
                 arch=None
                 ):
        super().__init__()
        self.arch = arch
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.depth = depth
        self.u = tq.TrainableUnitary(has_params=True,
                                     trainable=True,
                                     n_wires=self.n_wires - 1)

        self.rzs = tq.QuantumModuleList()
        for k in range(self.depth):
            self.rzs.append(tq.RZ())
        self.xcnot = tq.MultiXCNOT(n_wires=n_xcnot_wires)
        self.xcnot_wires = list(range(1, n_xcnot_wires)) + [0]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6)
        x = x.view(bsz, self.depth)
        x = F.tanh(x) * np.pi
        self.q_device.reset_states(bsz=bsz)

        # prepare for the |+> state
        tqf.h(self.q_device, wires=0)

        for k in range(self.depth):
            self.xcnot(self.q_device, wires=self.xcnot_wires)
            self.rzs[k](self.q_device, wires=0, params=x[:, k])
            self.xcnot(self.q_device, wires=self.xcnot_wires)
            self.u(self.q_device, wires=list(range(1, self.n_wires)),
                   inverse=(k % 2 == 1))

        x = self.measure(self.q_device)[:, :self.arch['output_len']]
        x = F.log_softmax(x, dim=1)
        x = x.squeeze()

        return x


class QFCModel5(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = 4
            self.encoder = tq.MultiPhaseEncoder(['rx'] * 4 + ['ry'] * 4 +
                                                ['rz'] * 4 + ['rx'] * 4)
            self.random_layer = tq.RandomLayer(
                n_ops=arch['n_random_ops'][0],
                wires=list(range(self.n_wires)))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x):
            self.q_device = q_device
            self.encoder(self.q_device, x)
            self.random_layer(self.q_device)

    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = x.view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process(self.q_device, self.q_layer, x)
        else:
            self.q_layer(self.q_device, x)
            x = self.measure(self.q_device)

        x = x[:, :self.arch['output_len']]
        x = F.log_softmax(x, dim=1)

        return x


class QFCModel5Resize4(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = 4
            self.encoder = tq.MultiPhaseEncoder(['rx'] * 4 + ['ry'] * 4 +
                                                ['rz'] * 4 + ['rx'] * 4)
            self.random_layer = tq.RandomLayer(
                n_ops=arch['n_random_ops'][0],
                wires=list(range(self.n_wires)))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x):
            self.q_device = q_device
            self.encoder(self.q_device, x)
            self.random_layer(self.q_device)

    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = x.view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process(self.q_device, self.q_layer, x)
        else:
            self.q_layer(self.q_device, x)
            x = self.measure(self.q_device)

        x = x[:, :self.arch['output_len']]
        x = F.log_softmax(x, dim=1)

        return x


class QFCModel6(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = 4
            self.encoder = tq.MultiPhaseEncoder(['rx'] * 4 + ['ry'] * 4 +
                                                ['rz'] * 4 + ['rx'] * 4)
            self.rx_layers = tq.QuantumModuleList()
            self.ry_layers = tq.QuantumModuleList()
            self.rz_layers = tq.QuantumModuleList()
            self.cnot_layers = tq.QuantumModuleList()

            for k in range(4):
                self.rx_layers.append(
                    tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.ry_layers.append(
                    tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.rz_layers.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.cnot_layers.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires,
                                    jump=1, circular=False))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x):
            self.q_device = q_device
            self.encoder(self.q_device, x)

            for k in range(4):
                self.rx_layers[k](self.q_device)
                self.ry_layers[k](self.q_device)
                self.rz_layers[k](self.q_device)
                self.cnot_layers[k](self.q_device)

    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        self.q_layer(self.q_device, x)

        x = self.measure(self.q_device)[:, :self.arch['output_len']]

        x = F.log_softmax(x, dim=1)

        return x


class QFCModel7(tq.QuantumModule):
    """difference: self.measure(self.q_device).reshape(bsz, 2, 2)"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = 4
            self.encoder = tq.MultiPhaseEncoder(['rx'] * 4 + ['ry'] * 4 +
                                                ['rz'] * 4 + ['rx'] * 4)
            self.rx_layers = tq.QuantumModuleList()
            self.ry_layers = tq.QuantumModuleList()
            self.rz_layers = tq.QuantumModuleList()
            self.cnot_layers = tq.QuantumModuleList()

            for k in range(4):
                self.rx_layers.append(
                    tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.ry_layers.append(
                    tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.rz_layers.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.cnot_layers.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires,
                                    jump=1, circular=False))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x):
            self.q_device = q_device
            self.encoder(self.q_device, x)

            for k in range(4):
                self.rx_layers[k](self.q_device)
                self.ry_layers[k](self.q_device)
                self.rz_layers[k](self.q_device)
                self.cnot_layers[k](self.q_device)

    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        self.q_layer(self.q_device, x)

        x = self.measure(self.q_device).reshape(bsz, 2, 2)
        x = x.sum(-1).squeeze()

        x = F.log_softmax(x, dim=1)

        return x


class QFCModel8(tq.QuantumModule):
    """difference: Op2QButterflyLayer"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = 4
            self.encoder = tq.MultiPhaseEncoder(['rx'] * 4 + ['ry'] * 4 +
                                                ['rz'] * 4 + ['rx'] * 4)
            self.rx_layers = tq.QuantumModuleList()
            self.ry_layers = tq.QuantumModuleList()
            self.rz_layers = tq.QuantumModuleList()
            self.cnot_layers = tq.QuantumModuleList()

            for k in range(4):
                self.rx_layers.append(
                    tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.ry_layers.append(
                    tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.rz_layers.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.cnot_layers.append(
                    tq.Op2QButterflyLayer(op=tq.CNOT,
                                          n_wires=self.n_wires))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x):
            self.q_device = q_device
            self.encoder(self.q_device, x)

            for k in range(4):
                self.rx_layers[k](self.q_device)
                self.ry_layers[k](self.q_device)
                self.rz_layers[k](self.q_device)
                self.cnot_layers[k](self.q_device)

    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        self.q_layer(self.q_device, x)

        x = self.measure(self.q_device).reshape(bsz, 2, 2)
        x = x.sum(-1).squeeze()

        x = F.log_softmax(x, dim=1)

        return x


class QFCModel9(tq.QuantumModule):
    """difference: Op2QDenseLayer"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = 4
            self.encoder = tq.MultiPhaseEncoder(['rx'] * 4 + ['ry'] * 4 +
                                                ['rz'] * 4 + ['rx'] * 4)
            self.rx_layers = tq.QuantumModuleList()
            self.ry_layers = tq.QuantumModuleList()
            self.rz_layers = tq.QuantumModuleList()
            self.cnot_layers = tq.QuantumModuleList()

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
                self.cnot_layers.append(
                    tq.Op2QDenseLayer(op=tq.CNOT,
                                      n_wires=self.n_wires))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x):
            self.q_device = q_device
            self.encoder(self.q_device, x)

            for k in range(self.arch['n_blocks']):
                self.rx_layers[k](self.q_device)
                self.ry_layers[k](self.q_device)
                self.rz_layers[k](self.q_device)
                self.cnot_layers[k](self.q_device)

    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        self.q_layer(self.q_device, x)

        x = self.measure(self.q_device).reshape(bsz, 2, 2)
        x = x.sum(-1).squeeze()

        x = F.log_softmax(x, dim=1)

        return x


class QFCModel10(tq.QuantumModule):
    """crx cry crz layers"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = 4
            if arch['encoder']['name'] == 'rxyzx':
                self.encoder = tq.MultiPhaseEncoder(['rx'] * 4 +
                                                    ['ry'] * 4 +
                                                    ['rz'] * 4 +
                                                    ['rx'] * 4)
            elif arch['encoder']['name'] == 'rxyzx_interleave':
                self.encoder = tq.MultiPhaseEncoder(
                    ['rx', 'ry', 'rz', 'rx'] * 4)
            elif arch['encoder']['name'] == 'ryzxy':
                self.encoder = tq.MultiPhaseEncoder(['ry'] * 4 +
                                                    ['rz'] * 4 +
                                                    ['rx'] * 4 +
                                                    ['ry'] * 4)
            elif arch['encoder']['name'] == 'rzxyz':
                self.encoder = tq.MultiPhaseEncoder(['rz'] * 4 +
                                                    ['rx'] * 4 +
                                                    ['ry'] * 4 +
                                                    ['rz'] * 4)
            elif arch['encoder']['name'] == 'u3u1':
                self.encoder = tq.MultiPhaseEncoder(
                    funcs=['u3', 'u1'] * 4,
                    wires=[0, 0, 1, 1, 2, 2, 3, 3]
                )
            elif arch['encoder']['name'] == 'u3rx':
                self.encoder = tq.MultiPhaseEncoder(
                    funcs=['u3', 'rx'] * 4,
                    wires=[0, 0, 1, 1, 2, 2, 3, 3]
                )
            elif arch['encoder']['name'] == 'u3ry':
                self.encoder = tq.MultiPhaseEncoder(
                    funcs=['u3', 'ry'] * 4,
                    wires=[0, 0, 1, 1, 2, 2, 3, 3]
                )
            elif arch['encoder']['name'] == 'u3rz':
                self.encoder = tq.MultiPhaseEncoder(
                    funcs=['u3', 'rz'] * 4,
                    wires=[0, 0, 1, 1, 2, 2, 3, 3]
                )
            else:
                raise NotImplementedError

            self.rx_layers = tq.QuantumModuleList()
            self.ry_layers = tq.QuantumModuleList()
            self.rz_layers = tq.QuantumModuleList()
            self.crx_layers = tq.QuantumModuleList()
            self.cry_layers = tq.QuantumModuleList()
            self.crz_layers = tq.QuantumModuleList()

            for k in range(arch['n_block']):
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
        def forward(self, q_device: tq.QuantumDevice, x):
            self.q_device = q_device
            self.encoder(self.q_device, x)

            for k in range(self.arch['n_block']):
                self.rx_layers[k](self.q_device)
                self.ry_layers[k](self.q_device)
                self.rz_layers[k](self.q_device)
                self.crx_layers[k](self.q_device)
                self.cry_layers[k](self.q_device)
                self.crz_layers[k](self.q_device)

    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        if self.arch['tanh']:
            x = F.tanh(x) * np.pi

        if use_qiskit:
            x = self.qiskit_processor.process(self.q_device, self.q_layer, x)
        else:
            self.q_layer(self.q_device, x)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x


class QFCModel11(tq.QuantumModule):
    """u3 and cu3 layers"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = 4
            self.encoder = tq.MultiPhaseEncoder(['rx'] * 4 + ['ry'] * 4 +
                                                ['rz'] * 4 + ['rx'] * 4)
            self.u3_0_layers = tq.QuantumModuleList()
            self.u3_1_layers = tq.QuantumModuleList()
            self.u3_2_layers = tq.QuantumModuleList()
            self.cu3_layers = tq.QuantumModuleList()

            for k in range(self.arch['n_block']):
                self.u3_0_layers.append(
                    tq.Op1QAllLayer(op=tq.U3, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.u3_1_layers.append(
                    tq.Op1QAllLayer(op=tq.U3, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.u3_2_layers.append(
                    tq.Op1QAllLayer(op=tq.U3, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.cu3_layers.append(
                    tq.Op2QAllLayer(op=tq.CU3, n_wires=self.n_wires,
                                    has_params=True, trainable=True,
                                    circular=True))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x):
            self.q_device = q_device
            self.encoder(self.q_device, x)

            for k in range(self.arch['n_block']):
                self.u3_0_layers[k](self.q_device)
                self.u3_1_layers[k](self.q_device)
                self.u3_2_layers[k](self.q_device)
                self.cu3_layers[k](self.q_device)

    def __init__(self, arch=None):
        super().__init__()
        self.arch = arch
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        self.q_layer(self.q_device, x)

        x = self.measure(self.q_device).reshape(bsz, 2, 2)
        x = x.sum(-1).squeeze()

        x = F.log_softmax(x, dim=1)

        return x


class QFCModel12(tq.QuantumModule):
    """crx cry crz layers"""
    class QLayer(tq.QuantumModule):
        def __init__(self, arch=None):
            super().__init__()
            self.arch = arch
            self.n_wires = 4
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
        self.q_layer = self.QLayer(arch=arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

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


class QFCModel13(QFCModel12):
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


class RandQLayer(tq.QuantumModule):
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


class QFCRandModel0(tq.QuantumModule):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.n_wires = arch['n_wires']
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            encoder_op_list_name_dict[arch['encoder_op_list_name']]
        )
        self.q_layer = RandQLayer(arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

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


class QVQERandModel0(tq.QuantumModule):
    def __init__(self, arch, hamil_info):
        super().__init__()
        self.arch = arch
        self.hamil_info = hamil_info
        self.n_wires = arch['n_wires']
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = RandQLayer(arch)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])

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
                                          device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x


class QVQEUCCSDModel0(tq.QuantumModule):
    def __init__(self, arch, hamil_info):
        super().__init__()
        self.arch = arch
        self.hamil_info = hamil_info
        self.n_wires = arch['n_wires']
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.molecule_data = molecule_name_dict[hamil_info['name']]
        self.molecule_data['transform'] = arch['transform']
        self.q_layer = generate_uccsd(self.molecule_data)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])

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
                                          device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x


model_dict = {
    'q_quanv0': QuanvModel0,
    'q_quanv1': QuanvModel1,
    'q_quanv2': QuanvModel2,
    'q_quanv3': QuanvModel3,
    'q_fc0': QFCModel0,
    'q_fc1': QFCModel1,
    'q_fc2': QFCModel2,
    'q_fc3': QFCModel3,
    'q_fc5': QFCModel5,
    'q_fc5_resize4': QFCModel5Resize4,
    'q_fc6': QFCModel6,
    'q_fc7': QFCModel7,
    'q_fc8': QFCModel8,
    'q_fc9': QFCModel9,
    'q_fc10': QFCModel10,
    'q_fc11': QFCModel11,
    'q_fc12': QFCModel12,
    'q_fc13': QFCModel13,
    'q_fc_rand0': QFCRandModel0,
    'vqe_rand0': QVQERandModel0,
    'vqe_uccsd': QVQEUCCSDModel0,
    'q_qsvt0': QSVT0,
}
