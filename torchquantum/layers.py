import torch.nn as nn
import torchquantum as tq
import numpy as np

from typing import Iterable
from torchquantum.plugins.qiskit_macros import QISKIT_INCOMPATIBLE_OPS
from torchpack.utils.logging import logger

__all__ = [
    'TrainableOpAll',
    'ClassicalInOpAll',
    'FixedOpAll',
    'TwoQAll',
    'RandomLayer',
    'RandomLayerAllTypes',
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
        assert self.n_gate == q_device.n_wires, \
            f"Number of rx gates ({self.n_gate}) is different from number " \
            f"of wires ({q_device.n_wires})!"

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
        assert self.n_gate == q_device.n_wires, \
            f"Number of rx gates ({self.n_gate}) is different from number " \
            f"of wires ({q_device.n_wires})!"

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
        assert self.n_gate == q_device.n_wires, \
            f"Number of rx gates ({self.n_gate}) is different from number " \
            f"of wires ({q_device.n_wires})!"

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
                 qiskit_comparible=False,
                 ):
        super().__init__()
        self.n_ops = n_ops
        self.wires = wires if isinstance(wires, Iterable) else [wires]
        self.n_wires = len(wires)

        op_types = op_types if isinstance(op_types, Iterable) else [op_types]
        op_ratios = op_ratios if isinstance(op_ratios, Iterable) else [
            op_ratios]
        op_types_valid = []
        op_ratios_valid = []

        if qiskit_comparible:
            for op_type, op_ratio in zip(op_types, op_ratios):
                if op_type in QISKIT_INCOMPATIBLE_OPS:
                    logger.warning(f"Remove {op_type} from op_types to make "
                                   f"the layer qiskit-compatible.")
                else:
                    op_types_valid.append(op_type)
                    op_ratios_valid.append(op_ratio)
        else:
            op_types_valid = op_types
            op_ratios_valid = op_ratios

        self.op_types = op_types_valid
        self.op_ratios = np.array(op_ratios_valid) / sum(op_ratios_valid)

        self.seed = seed
        self.op_list = tq.QuantumModuleList()
        if seed is not None:
            np.random.seed(seed)
        self.build_random_layer()

    def build_random_layer(self):
        cnt = 0
        while cnt < self.n_ops:
            op = np.random.choice(self.op_types, p=self.op_ratios)
            n_op_wires = op.num_wires
            if n_op_wires > self.n_wires:
                continue
            if n_op_wires == -1:
                is_AnyWire = True
                n_op_wires = self.n_wires
            else:
                is_AnyWire = False

            op_wires = list(np.random.choice(self.wires, size=n_op_wires,
                                             replace=False))
            if is_AnyWire:
                if op().name in ['MultiRZ']:
                    operation = op(has_params=True, trainable=True,
                        n_wires=n_op_wires, wires=op_wires)
                else:
                    operation = op(n_wires=n_op_wires, wires=op_wires)
            elif op().name in tq.Operator.parameterized_ops:
                operation = op(has_params=True, trainable=True, wires=op_wires)
            else:
                operation = op(wires=op_wires)
            self.op_list.append(operation)
            cnt += 1

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for op in self.op_list:
            op(q_device)


class RandomLayerAllTypes(RandomLayer):
    def __init__(self,
                 n_ops,
                 wires,
                 op_ratios=None,
                 op_types=(tq.Hadamard,
                           tq.PauliX,
                           tq.PauliY,
                           tq.PauliZ,
                           tq.S,
                           tq.T,
                           tq.SX,
                           tq.CNOT,
                           tq.CZ,
                           tq.CY,
                           tq.RX,
                           tq.RY,
                           tq.RZ,
                           tq.SWAP,
                           tq.CSWAP,
                           tq.Toffoli,
                           tq.PhaseShift,
                           tq.Rot,
                           tq.MultiRZ,
                           tq.CRX,
                           tq.CRY,
                           tq.CRZ,
                           tq.CRot,
                           tq.U1,
                           tq.U2,
                           tq.U3,
                           tq.MultiCNOT,
                           tq.MultiXCNOT,
                           ),
                 seed=None,
                 qiskit_comparible=False,
                 ):
        if op_ratios is None:
            op_ratios = [1] * len(op_types)
        super().__init__(
            n_ops,
            wires,
            op_ratios,
            op_types,
            seed,
            qiskit_comparible,
        )


class SimpleQLayer(tq.QuantumModule):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires
        self.gate1 = tq.RX(has_params=True, trainable=True)
        self.gate2 = tq.RY(has_params=True, trainable=True)
        self.gate3 = tq.RZ(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, q_dev):
        self.q_device = q_dev
        tqf.x(q_dev, wires=0, static=self.static_mode,
              parent_graph=self.graph)
        self.gate1(q_dev, wires=1)
        self.gate2(q_dev, wires=1)
        self.gate3(q_dev, wires=1)
        tqf.x(q_dev, wires=2, static=self.static_mode,
              parent_graph=self.graph)
