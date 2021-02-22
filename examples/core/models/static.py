import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import functools

from abc import ABCMeta

__all__ = ['Static']


def register_graph(f):
    @functools.wraps(f)
    def forward_register_graph(*args, **kwargs):
        if args[0].static_mode and args[0].parent_graph is not None:
            args[0].parent_graph.add_op(args[0])
        res = f(*args, **kwargs)
        return res
    return forward_register_graph


class TQAll(tq.QuantumModule):
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.n_gate = n_gate
        self.op = op()

    @register_graph
    def forward(self, q_device: tq.QuantumDevice):
        for k in range(self.n_gate-1):
            self.op(q_device, wires=[k, k + 1])
        self.op(q_device, wires=[self.n_gate-1, 0])


class OpAll(tq.QuantumModule):
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.submodules = tq.QuantumModuleList()
        self.q_layer0 = TQAll(n_gate, tq.CNOT)
        self.n_gate = n_gate
        for k in range(self.n_gate):
            self.submodules.append(op())

    @register_graph
    def forward(self, q_device: tq.QuantumDevice, x):
        for k in range(self.n_gate):
            self.submodules[k](q_device, wires=k, params=x[:, k])
            tqf.rx(q_device, wires=0, params=x[:, 0],
                   static=self.static_mode, graph=self.graph)
        self.q_layer0(q_device)


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
        self.q_device0 = tq.QuantumDevice(n_wire=10)
        self.q_layer0 = tq.TrainableOpAll(n_gate=10, op=tq.RX)
        self.q_layer1 = OpAll(n_gate=10, op=tq.RY)
        self.q_layer2 = tq.RX(has_params=True,
                              trainable=False,
                              init_params=-np.pi / 4)

        self.q_layer3 = tq.RZ(has_params=True,
                              trainable=True)
        self.q_device1 = tq.QuantumDevice(n_wire=3)
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

    def forward(self, x):
        self.q_layer1.static_on()
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

        x = tq.expval(self.q_device0, list(range(10)), [tq.PauliY()] * 10)

        output = F.log_softmax(x, dim=1)

        return output
