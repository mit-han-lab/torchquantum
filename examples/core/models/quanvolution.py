import torchquantum as tq
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Quanvolution']


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

        self.q_device0 = tq.QuantumDevice(n_wire=9)
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
        self.q_device0 = tq.QuantumDevice(n_wire=9)
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
