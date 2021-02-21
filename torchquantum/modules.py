import torch.nn as nn
import torchquantum as tq


__all__ = [
    'QuantumModule',
    'TrainableOpAll',
    'ClassicalInOpAll',
    'FixedOpAll',
    'TwoQAll',
]


class QuantumModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class TrainableOpAll(QuantumModule):
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


class ClassicalInOpAll(QuantumModule):
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


class FixedOpAll(QuantumModule):
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


class TwoQAll(QuantumModule):
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.n_gate = n_gate
        self.op = op()

    def forward(self, q_device: tq.QuantumDevice):
        for k in range(self.n_gate-1):
            self.op(q_device, wires=[k, k + 1])
        self.op(q_device, wires=[self.n_gate-1, 0])


def test():
    pass


if __name__ == '__main__':
    test()
