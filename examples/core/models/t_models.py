import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import functools
from torchquantum.utils import Timer

__all__ = ['Static', 'Quanvolution', 'Hybrid', 'model_dict']


class TQAll(tq.QuantumModule):
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.submodules = tq.QuantumModuleList()
        self.n_gate = n_gate
        for k in range(self.n_gate):
            self.submodules.append(op())

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for k in range(self.n_gate - 1):
            self.submodules[k](q_device, wires=[k, k + 1])
        self.submodules[-1](q_device, wires=[self.n_gate - 1, 0])


class OpAll(tq.QuantumModule):
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.submodules = tq.QuantumModuleList()
        self.q_layer0 = TQAll(n_gate, tq.CNOT)
        self.n_gate = n_gate
        for k in range(self.n_gate):
            self.submodules.append(op())

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        # tqf.rx(q_device, wires=6, params=x[:, 0],
        #        static=self.static_mode, parent_graph=self.graph)
        for k in range(self.n_gate):
            self.submodules[k](q_device, wires=k, params=x[:, k%10])
            tqf.rx(q_device, wires=0, params=x[:, 0],
                   static=self.static_mode, parent_graph=self.graph)
        self.q_layer0(q_device)


class TestModule(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_gate = 12
        self.gate0 = tq.CNOT()
        self.gate1 = tq.CNOT()
        self.submodules = tq.QuantumModuleList()
        self.q_layer0 = TQAll(self.n_gate, tq.CNOT)
        for k in range(self.n_gate):
            self.submodules.append(tq.RY())
        # for k in range(self.n_gate):
        #     self.submodules.append(tq.CNOT())
        # self.gate0 = tq.RY(has_params=False, trainable=False)
        # self.gate1 = tq.RX(has_params=False, trainable=False)
        # self.gate2 = tq.RZ(has_params=False, trainable=False)
        self.gate1 = tq.RX(has_params=True, trainable=True)
        self.gate2 = tq.RZ(has_params=True, trainable=True)
        self.gate3 = tq.RY(has_params=True, trainable=True)
        # self.gate3 = tq.CNOT()
        self.gate4 = tq.RX(has_params=True, trainable=True)
        self.gate5 = tq.RZ(has_params=True, trainable=True)
        self.gate6 = tq.RY(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        self.gate1(q_device, wires=3)
        self.gate2(q_device, wires=4)
        self.gate3(q_device, wires=3)
        self.gate4(q_device, wires=3)
        self.gate5(q_device, wires=3)
        self.gate6(q_device, wires=3)

        # for k in range(self.n_gate):
        #     self.submodules[k](q_device, wires=k, params=x[:, k%10])
        #     tqf.rx(q_device, wires=k, params=x[:, 0],
        #            static=self.static_mode, parent_graph=self.graph)
        #     tqf.cnot(q_device, wires=[k, (k + 1) % self.n_gate],
        #              static=self.static_mode, parent_graph=self.graph)
        #     tqf.rx(q_device, wires=k, params=x[:, 0],
        #            static=self.static_mode, parent_graph=self.graph)
        #     tqf.cnot(q_device, wires=[(k + 1) % self.n_gate, k],
        #              static=self.static_mode, parent_graph=self.graph)
        #     tqf.rx(q_device, wires=k, params=x[:, 0],
        #            static=self.static_mode, parent_graph=self.graph)
        #     tqf.cnot(q_device, wires=[k, (k + 1) % self.n_gate],
        #              static=self.static_mode, parent_graph=self.graph)
        #     tqf.cnot(q_device, wires=[(k + 1) % self.n_gate, k],
        #              static=self.static_mode, parent_graph=self.graph)
        #     tqf.rx(q_device, wires=k, params=x[:, 0],
        #            static=self.static_mode, parent_graph=self.graph)
        #     tqf.cnot(q_device, wires=[k, (k + 1) % self.n_gate],
        #              static=self.static_mode, parent_graph=self.graph)
        #     tqf.cnot(q_device, wires=[(k + 1) % self.n_gate, k],
        #              static=self.static_mode, parent_graph=self.graph)
        #
        # self.q_layer0(q_device)

        # self.gate0(q_device, wires=[7, 4])
        # self.gate1(q_device, wires=[3, 9])

        # self.gate0(q_device, wires=1, params=x[:, 2])
        # self.gate1(q_device, wires=5, params=x[:, 0])
        # self.gate2(q_device, wires=7, params=x[:, 6])


        # self.gate2(q_device, wires=5)
        # self.gate3(q_device, wires=[3, 5])
        # self.gate4(q_device, wires=5)


class Static(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.sigmoid = nn.Sigmoid()
        self.q_device0 = tq.QuantumDevice(n_wires=12)
        self.q_device0_1 = tq.QuantumDevice(n_wires=10)
        self.q_layer0 = tq.TrainableOpAll(n_gate=12, op=tq.RX)
        # self.q_layer1 = OpAll(n_gate=10, op=tq.RY)
        self.q_layer2 = tq.RX(has_params=True,
                              trainable=False,
                              init_params=-np.pi / 4)

        self.q_layer3 = tq.RZ(has_params=True,
                              trainable=True)
        self.q_device1 = tq.QuantumDevice(n_wires=3)
        self.q_layer4 = tq.CY()
        self.q_layer5 = tq.Toffoli()
        self.q_layer6 = tq.PhaseShift(has_params=True,
                                      trainable=True)
        self.q_layer7 = tq.Rot(has_params=True,
                               trainable=True)
        self.q_layer8 = tq.MultiRZ(has_params=True,
                                   trainable=True,
                                   n_wires=5)
        self.q_layer9 = tq.CRX(has_params=True,
                               trainable=True)
        self.q_layer10 = tq.CRY(has_params=True,
                                trainable=True)
        self.q_layer11 = tq.CRZ(has_params=True,
                                trainable=True)
        self.q_layer12 = tq.CRot(has_params=True,
                                 trainable=False,
                                 init_params=[-np.pi / 4,
                                              np.pi / 4,
                                              np.pi / 2])
        self.q_layer13 = tq.U1(has_params=True,
                               trainable=False,
                               init_params=np.pi/7)
        self.q_layer14 = tq.U2(has_params=True,
                               trainable=True,
                               init_params=[np.pi/7, np.pi/8.8])
        self.q_layer15 = tq.U3(has_params=True,
                               trainable=True)
        self.q_layer16 = tq.QubitUnitary(has_params=True,
                                         trainable=False,
                                         init_params=[[1, 0], [0, 1]])
        self.q_test_layer = TestModule()
        self.random_layer = tq.RandomLayer(30, wires=[0, 3, 5, 7])
        # self.random_layer.static_on(wires_per_block=3)

        # self.q_test_layer.static_on(wires_per_block=4)

    def forward(self, x):
        # self.q_layer1.static_on(wires_per_block=3)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * 2 * np.pi

        self.q_device0.reset_states(x.shape[0])
        # self.q_device1.reset_states(x.shape[0])
        # self.q_device0_1.reset_states(x.shape[0])
        self.q_layer0(self.q_device0)
        self.random_layer(self.q_device0)
        # self.q_layer0(self.q_device0_1)

        # with Timer('gpu', 'static', 50):
        #     for _ in range(500):
        # self.q_test_layer.static_on(wires_per_block=4)
        self.q_test_layer(self.q_device0, x)
        # self.q_test_layer.static_off()

        # self.q_test_layer.static_off()
        # with Timer('gpu', 'dynamic', 50):
        #     for _ in range(500):
        #         self.q_test_layer(self.q_device0, x)
        # self.q_test_layer(self.q_device0_1, x)
        # dif = abs(self.q_device0.states-self.q_device0_1.states)

        # print(dif.max(), dif.mean())
        # exit(0)

        # self.q_layer1(self.q_device0, x)
        # self.q_layer1.static_off()
        # self.q_layer1(self.q_device0_1, x)

        # tqf.rx(self.q_device0, 1, x[:, 1])
        # self.q_layer2(self.q_device0, wires=5)
        # tqf.ry(self.q_device0, 2, x[:, 2])
        # tqf.rz(self.q_device0, 3, x[:, 3])
        # tqf.s(self.q_device0, 4)
        # tqf.t(self.q_device0, 5)
        # self.q_layer3(self.q_device0, wires=6)
        # tqf.sx(self.q_device0, 7)
        # tqf.x(self.q_device1, wires=0)
        # tqf.cnot(self.q_device1, wires=[0, 2])
        # tqf.cnot(self.q_device1, wires=[0, 1])
        # tqf.cnot(self.q_device1, wires=[2, 0])
        # tqf.cz(self.q_device0, wires=[0, 5])
        # tqf.cnot(self.q_device0, wires=[0, 5])
        # tqf.cy(self.q_device0, wires=[0, 5])
        # self.q_layer4(self.q_device0, wires=[3, 8])
        # tqf.swap(self.q_device0, wires=[2, 3])
        # tqf.cswap(self.q_device0, wires=[4, 5, 6])
        # self.q_layer5(self.q_device0, wires=[8, 5, 0])
        # self.q_layer6(self.q_device0, wires=8)
        # tqf.phaseshift(self.q_device0, 7, x[:, 7])
        # self.q_layer7(self.q_device0, wires=4)
        # tqf.rot(self.q_device0, 5, x[:, 6:9])
        # self.q_layer8(self.q_device0, wires=[2, 3, 4, 5, 6])
        # tqf.multirz(self.q_device0, wires=[3, 4, 6, 7, 8], params=x[:, 5],
        #             n_wires=5)
        # tqf.crx(self.q_device0, wires=[0, 1], params=x[:, 6])
        # self.q_layer9(self.q_device0, wires=[4, 5])
        # tqf.cry(self.q_device0, wires=[0, 1], params=x[:, 6])
        # self.q_layer10(self.q_device0, wires=[4, 5])
        # tqf.crz(self.q_device0, wires=[0, 1], params=x[:, 6])
        # self.q_layer11(self.q_device0, wires=[4, 5])
        # self.q_layer12(self.q_device0, wires=[5, 6])
        # tqf.crot(self.q_device0, wires=[7, 8], params=x[:, 5:8])
        # self.q_layer13(self.q_device0, wires=1)
        # tqf.u1(self.q_device0, wires=2, params=x[:, 9])
        # self.q_layer14(self.q_device0, wires=8)
        # tqf.u2(self.q_device0, wires=4, params=x[:, 0:2])
        # self.q_layer15(self.q_device0, wires=5)
        # tqf.u3(self.q_device0, wires=5, params=x[: 4:7])
        # self.q_layer16(self.q_device0, wires=3)
        # tqf.qubitunitary(self.q_device0, wires=[1, 2], params=[[1, 0, 0, 0],
        #                                                        [0, 1, 0, 0],
        #                                                        [0, 0, 0, 1],
        #                                                        [0, 0, 1, 0]])

        x = tq.expval(self.q_device0, list(range(10)), [tq.PauliY()] * 10)

        output = F.log_softmax(x, dim=1)

        return output


class QLayer0(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_gate = 9

        self.h_all0 = tq.FixedOpAll(n_gate=9, op=tq.Hadamard)
        self.s_all0 = tq.FixedOpAll(n_gate=9, op=tq.S)
        self.t_all0 = tq.FixedOpAll(n_gate=9, op=tq.T)

        self.rx_all0 = tq.ClassicalInOpAll(n_gate=9, op=tq.RX)
        self.ry_all0 = tq.ClassicalInOpAll(n_gate=9, op=tq.RY)
        self.rz_all0 = tq.ClassicalInOpAll(n_gate=9, op=tq.RY)

        self.trainable_rx_all0 = tq.TrainableOpAll(n_gate=9, op=tq.RX)
        self.trainable_ry_all0 = tq.TrainableOpAll(n_gate=9, op=tq.RY)
        self.trainable_rz_all0 = tq.TrainableOpAll(n_gate=9, op=tq.RZ)
        self.trainable_rx_all1 = tq.TrainableOpAll(n_gate=9, op=tq.RX)
        self.trainable_ry_all1 = tq.TrainableOpAll(n_gate=9, op=tq.RY)
        self.trainable_rz_all1 = tq.TrainableOpAll(n_gate=9, op=tq.RZ)

        self.q_device0 = tq.QuantumDevice(n_wires=9)
        self.cnot_all = tq.TwoQAll(n_gate=9, op=tq.CNOT)
        self.cz_all = tq.TwoQAll(n_gate=9, op=tq.CZ)
        self.cy_all = tq.TwoQAll(n_gate=9, op=tq.CY)

    def forward(self, x):
        self.q_device0.reset_states(x.shape[0])
        self.h_all0(self.q_device0)

        self.trainable_rx_all0(self.q_device0)
        self.cnot_all(self.q_device0)
        self.rx_all0(self.q_device0, x)

        self.trainable_ry_all0(self.q_device0)
        self.cnot_all(self.q_device0)
        self.ry_all0(self.q_device0, x)

        self.trainable_rz_all0(self.q_device0)
        self.cnot_all(self.q_device0)
        self.rz_all0(self.q_device0, x)

        self.t_all0(self.q_device0)

        self.trainable_rx_all1(self.q_device0)
        self.cz_all(self.q_device0)

        self.trainable_ry_all1(self.q_device0)
        self.cy_all(self.q_device0)

        self.trainable_rz_all0(self.q_device0)

        self.s_all0(self.q_device0)

        x = tq.expval(self.q_device0, list(range(9)), [tq.PauliZ()] * 9)

        return x


class QLayer1(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_gate = 9
        self.h_all0 = tq.FixedOpAll(n_gate=9, op=tq.Hadamard)
        self.rx_all0 = tq.ClassicalInOpAll(n_gate=9, op=tq.RX)
        self.trainable_rx_all0 = tq.TrainableOpAll(n_gate=9, op=tq.RX)
        self.ry_all0 = tq.ClassicalInOpAll(n_gate=9, op=tq.RY)
        self.trainable_ry_all0 = tq.TrainableOpAll(n_gate=9, op=tq.RY)
        self.rz_all0 = tq.ClassicalInOpAll(n_gate=9, op=tq.RZ)
        self.trainable_rz_all0 = tq.TrainableOpAll(n_gate=9, op=tq.RZ)
        self.q_device0 = tq.QuantumDevice(n_wires=9)
        self.cnot_all = tq.TwoQAll(n_gate=9, op=tq.CNOT)

    def forward(self, x):
        self.q_device0.reset_states(x.shape[0])
        self.h_all0(self.q_device0)

        self.rx_all0(self.q_device0, x[:, 0, :])
        self.cnot_all(self.q_device0)
        self.trainable_rx_all0(self.q_device0)

        self.ry_all0(self.q_device0, x[:, 1, :])
        self.cnot_all(self.q_device0)
        self.trainable_ry_all0(self.q_device0)

        self.rz_all0(self.q_device0, x[:, 2, :])
        self.cnot_all(self.q_device0)
        self.trainable_rz_all0(self.q_device0)

        x = tq.expval(self.q_device0, list(range(9)), [tq.PauliZ()] * 9)

        return x


class Quanvolution(tq.QuantumModule):
    """
    Convolution with quantum filter
    """
    def __init__(self):
        super().__init__()
        self.n_gate = 9
        self.q_layer0 = QLayer0()
        # self.q_layer1 = QLayer1()
        self.conv0 = torch.nn.Conv2d(9, 3, 1, 1)
        self.conv1 = torch.nn.Conv2d(9, 1, 1, 1)
        self.fc0 = nn.Linear(7 * 7, 10)
        # self.fc1 = nn.Linear(13 * 13, 10)
        # self.fc1 = nn.Linear(13 * 13, 10)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.max_pool2d(x, 3)

        # out0 = torch.empty([x.shape[0], 9, 26, 26]).to(x)

        # need to change from conv 2d dim to batch dim to increase speed!
        q_layer0_in = torch.empty([x.shape[0], 7, 7, 9])

        for j in range(0, 7):
            for k in range(0, 7):
                q_layer0_in[:, j, k, :] = \
                    torch.stack([
                        x[:, 0, j, k],
                        x[:, 0, j, k + 1],
                        x[:, 0, j, k + 2],
                        x[:, 0, j + 1, k],
                        x[:, 0, j + 1, k + 1],
                        x[:, 0, j + 1, k + 2],
                        x[:, 0, j + 1, k],
                        x[:, 0, j + 1, k + 1],
                        x[:, 0, j + 1, k + 2]
                    ], dim=-1)
                # out0[:, :, j, k] = q_results
        q_layer0_in = q_layer0_in.view(bsz * 7 * 7, 9)
        q_layer0_out = self.q_layer0(q_layer0_in)
        q_layer0_out = q_layer0_out.view(bsz, 7, 7, 9).permute(0, 3, 1, 2)

        # x = self.conv1(q_layer0_out)
        x = torch.sum(q_layer0_out, dim=-3)

        # out1 = torch.empty([x.shape[0], 9, 24, 24]).to(x)
        # for j in range(0, 24):
        #     for k in range(0, 24):
        #         q_results = self.q_layer1(
        #             torch.stack([
        #                 x[:, :, j, k],
        #                 x[:, :, j, k + 1],
        #                 x[:, :, j, k + 2],
        #                 x[:, :, j + 1, k],
        #                 x[:, :, j + 1, k + 1],
        #                 x[:, :, j + 1, k + 2],
        #                 x[:, :, j + 1, k],
        #                 x[:, :, j + 1, k + 1],
        #                 x[:, :, j + 1, k + 2]
        #             ], dim=-1)
        #         )
        #         out1[:, :, j, k] = q_results

        # x = self.conv1(out0)
        # x = F.max_pool2d(x, 2, padding=1)
        x = torch.flatten(x, 1)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc0(x)
        # x = x[:, :10]

        output = F.log_softmax(x, dim=1)

        return output


class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.sigmoid = nn.Sigmoid()
        self.q_device0 = tq.QuantumDevice(n_wires=10)
        self.q_layer0 = tq.TrainableOpAll(n_gate=10, op=tq.RX)
        self.q_layer1 = tq.ClassicalInOpAll(n_gate=10, op=tq.RX)
        self.q_layer2 = tq.RX(has_params=True,
                              trainable=False,
                              init_params=-np.pi / 4)

        self.q_layer3 = tq.RZ(has_params=True,
                              trainable=True)
        self.q_device1 = tq.QuantumDevice(n_wires=3)
        self.q_layer4 = tq.CY()
        self.q_layer5 = tq.Toffoli()
        self.q_layer6 = tq.PhaseShift(has_params=True,
                                      trainable=True)
        self.q_layer7 = tq.Rot(has_params=True,
                               trainable=True)
        self.q_layer8 = tq.MultiRZ(has_params=True,
                                   trainable=True,
                                   n_wires=5)
        self.q_layer9 = tq.CRX(has_params=True,
                               trainable=True)
        self.q_layer10 = tq.CRY(has_params=True,
                                trainable=True)
        self.q_layer11 = tq.CRZ(has_params=True,
                                trainable=True)
        self.q_layer12 = tq.CRot(has_params=True,
                                 trainable=False,
                                 init_params=[-np.pi / 4,
                                              np.pi / 4,
                                              np.pi / 2])
        self.q_layer13 = tq.U1(has_params=True,
                               trainable=False,
                               init_params=np.pi/7)
        self.q_layer14 = tq.U2(has_params=True,
                               trainable=True,
                               init_params=[np.pi/7, np.pi/8.8])
        self.q_layer15 = tq.U3(has_params=True,
                               trainable=True)
        self.q_layer16 = tq.QubitUnitary(has_params=True,
                                         trainable=False,
                                         init_params=[[1, 0], [0, 1]])
        self.q_layer17 = tq.MultiCNOT(n_wires=5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * 2 * np.pi

        self.q_device0.reset_states(x.shape[0])
        self.q_device1.reset_states(x.shape[0])
        self.q_layer0(self.q_device0)
        self.q_layer1(self.q_device0, x)
        tqf.rx(self.q_device0, 1, x[:, 1])
        self.q_layer2(self.q_device0, wires=5)
        tqf.ry(self.q_device0, 2, x[:, 2])
        tqf.rz(self.q_device0, 3, x[:, 3])
        tqf.s(self.q_device0, 4)
        tqf.t(self.q_device0, 5)
        self.q_layer3(self.q_device0, wires=6)
        tqf.sx(self.q_device0, 7)
        tqf.x(self.q_device1, wires=0)
        tqf.cnot(self.q_device1, wires=[0, 2])
        tqf.cnot(self.q_device1, wires=[0, 1])
        tqf.cnot(self.q_device1, wires=[2, 0])
        tqf.cz(self.q_device0, wires=[0, 5])
        tqf.cnot(self.q_device0, wires=[0, 5])
        tqf.cy(self.q_device0, wires=[0, 5])
        self.q_layer4(self.q_device0, wires=[3, 8])
        tqf.swap(self.q_device0, wires=[2, 3])
        tqf.cswap(self.q_device0, wires=[4, 5, 6])
        self.q_layer5(self.q_device0, wires=[8, 5, 0])
        self.q_layer6(self.q_device0, wires=8)
        tqf.phaseshift(self.q_device0, 7, x[:, 7])
        self.q_layer7(self.q_device0, wires=4)
        tqf.rot(self.q_device0, 5, x[:, 6:9])
        self.q_layer8(self.q_device0, wires=[2, 3, 4, 5, 6])
        tqf.multirz(self.q_device0, wires=[3, 4, 6, 7, 8], params=x[:, 5],
                    n_wires=5)
        tqf.crx(self.q_device0, wires=[0, 1], params=x[:, 6])
        self.q_layer9(self.q_device0, wires=[4, 5])
        tqf.cry(self.q_device0, wires=[0, 1], params=x[:, 6])
        self.q_layer10(self.q_device0, wires=[4, 5])
        tqf.crz(self.q_device0, wires=[0, 1], params=x[:, 6])
        self.q_layer11(self.q_device0, wires=[4, 5])
        self.q_layer12(self.q_device0, wires=[5, 6])
        tqf.crot(self.q_device0, wires=[7, 8], params=x[:, 5:8])
        self.q_layer13(self.q_device0, wires=1)
        tqf.u1(self.q_device0, wires=2, params=x[:, 9])
        self.q_layer14(self.q_device0, wires=8)
        tqf.u2(self.q_device0, wires=4, params=x[:, 0:2])
        self.q_layer15(self.q_device0, wires=5)
        tqf.u3(self.q_device0, wires=5, params=x[: 4:7])
        self.q_layer16(self.q_device0, wires=3)
        tqf.qubitunitary(self.q_device0, wires=[1, 2], params=[[1, 0, 0, 0],
                                                               [0, 1, 0, 0],
                                                               [0, 0, 0, 1],
                                                               [0, 0, 1, 0]])
        tqf.multicnot(self.q_device0, wires=[0, 1, 2, 3], n_wires=4)

        self.q_layer17(self.q_device0, wires=[8, 4, 3, 2, 9])

        x = tq.expval(self.q_device0, list(range(10)), [tq.PauliY()] * 10)

        output = F.log_softmax(x, dim=1)

        return output


model_dict = {
    't_static': Static,
    't_quanv': Quanvolution,
    't_hybrid': Hybrid
}
