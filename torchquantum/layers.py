import torch.nn as nn
import torchquantum as tq
import numpy as np

__all__ = [
    'TrainableOpAll',
    'ClassicalInOpAll',
    'FixedOpAll',
    'TwoQAll',
    'RandomLayer'
]


class TrainableOpAll(tq.QuantumModule):
    """Rotation rx on all qubits
    The rotation angle is a parameter of each rotation gate
    One potential optimization is to compute the unitary of all gates
    together.
    """
    def __init__(self, n_gate: int, op: tq.Operation):
        super().__init__()
        self.n_gate = n_gate
        self.gate_all = nn.ModuleList()
        for k in range(self.n_gate):
            self.gate_all.append(op(
                has_params=True,
                trainable=True))

    def forward(self, q_device: tq.QuantumDevice):
        # rx on all wires, assert the number of gate is the same as the number
        # of wires in the device.
        assert self.n_gate == q_device.n_wire, \
            f"Number of rx gates ({self.n_gate}) is different from number " \
            f"of wires ({q_device.n_wire})!"

        for k in range(self.n_gate):
            self.gate_all[k](q_device, wires=k)


class ClassicalInOpAll(tq.QuantumModule):
    """Rotation rx on all qubits
    The rotation angle is from input activation
    """
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.n_gate = n_gate
        self.gate_all = nn.ModuleList()
        for k in range(self.n_gate):
            self.gate_all.append(op())

    def forward(self, q_device: tq.QuantumDevice, x):
        # rx on all wires, assert the number of gate is the same as the number
        # of wires in the device.
        assert self.n_gate == q_device.n_wire, \
            f"Number of rx gates ({self.n_gate}) is different from number " \
            f"of wires ({q_device.n_wire})!"

        for k in range(self.n_gate):
            self.gate_all[k](q_device, wires=k, params=x[:, k])


class FixedOpAll(tq.QuantumModule):
    """Rotation rx on all qubits
    The rotation angle is from input activation
    """
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.n_gate = n_gate
        self.gate_all = nn.ModuleList()
        for k in range(self.n_gate):
            self.gate_all.append(op())

    def forward(self, q_device: tq.QuantumDevice):
        # rx on all wires, assert the number of gate is the same as the number
        # of wires in the device.
        assert self.n_gate == q_device.n_wire, \
            f"Number of rx gates ({self.n_gate}) is different from number " \
            f"of wires ({q_device.n_wire})!"

        for k in range(self.n_gate):
            self.gate_all[k](q_device, wires=k)


class TwoQAll(tq.QuantumModule):
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.n_gate = n_gate
        self.op = op()

    def forward(self, q_device: tq.QuantumDevice):
        for k in range(self.n_gate-1):
            self.op(q_device, wires=[k, k + 1])
        self.op(q_device, wires=[self.n_gate-1, 0])


class RandomLayer(tq.QuantumModule):
    def __init__(self,
                 n_ops,
                 wires,
                 op_ratios=(1, 1, 1, 1),
                 op_types=(tq.RX, tq.RY, tq.RZ, tq.CNOT),
                 seed=None,
                 ):
        super().__init__()
        self.n_ops = n_ops
        self.wires = wires
        self.n_wires = len(wires)
        self.op_ratios = np.array(op_ratios) / sum(op_ratios)
        self.op_types = op_types
        self.seed = seed
        self.op_list = []
        if seed is None:
            np.random.seed(42)
        else:
            np.random.seed(seed)
        self.build_random_layer()

    def build_random_layer(self):
        for _ in range(self.n_ops):
            op = np.random.choice(self.op_types, p=self.op_ratios)
            n_op_wires = op.num_wires
            op_wires = list(np.random.choice(self.wires, size=n_op_wires,
                                             replace=False))
            if op().name in tq.Operator.parameterized_ops:
                operation = op(has_params=True, trainable=True)
            else:
                operation = op()
            self.op_list.append({'wires': op_wires, 'op': operation})

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for pair in self.op_list:
            pair['op'](q_device, wires=pair['wires'])
