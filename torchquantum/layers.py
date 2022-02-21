import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np


from typing import Iterable
from torchquantum.plugins.qiskit_macros import QISKIT_INCOMPATIBLE_FUNC_NAMES
from torchpack.utils.logging import logger

__all__ = [
    'QuantumModuleFromOps',
    'TrainableOpAll',
    'ClassicalInOpAll',
    'FixedOpAll',
    'TwoQAll',
    'RandomLayer',
    'RandomLayerAllTypes',
    'Op1QAllLayer',
    'Op2QAllLayer',
    'Op2QButterflyLayer',
    'Op2QDenseLayer',
    'layer_name_dict',
    'CXLayer',
    'CXCXCXLayer',
    'SWAPSWAPLayer',
    'RXYZCXLayer0',
]


class QuantumModuleFromOps(tq.QuantumModule):
    def __init__(self, ops):
        super().__init__()
        self.ops = tq.QuantumModuleList(ops)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for op in self.ops:
            op(q_device)


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

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        # rx on all wires, assert the number of gate is the same as the number
        # of wires in the device.
        assert self.n_gate == q_device.n_wires, \
            f"Number of rx gates ({self.n_gate}) is different from number " \
            f"of wires ({q_device.n_wires})!"

        for k in range(self.n_gate):
            self.gate_all[k](q_device, wires=k)


class ClassicalInOpAll(tq.QuantumModule):
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.n_gate = n_gate
        self.gate_all = nn.ModuleList()
        for k in range(self.n_gate):
            self.gate_all.append(op())

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        # rx on all wires, assert the number of gate is the same as the number
        # of wires in the device.
        assert self.n_gate == q_device.n_wires, \
            f"Number of rx gates ({self.n_gate}) is different from number " \
            f"of wires ({q_device.n_wires})!"

        for k in range(self.n_gate):
            self.gate_all[k](q_device, wires=k, params=x[:, k])


class FixedOpAll(tq.QuantumModule):
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.n_gate = n_gate
        self.gate_all = nn.ModuleList()
        for k in range(self.n_gate):
            self.gate_all.append(op())

    @tq.static_support
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

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        for k in range(self.n_gate-1):
            self.op(q_device, wires=[k, k + 1])
        self.op(q_device, wires=[self.n_gate-1, 0])


class RandomLayer(tq.QuantumModule):
    def __init__(self,
                 wires,
                 n_ops=None,
                 n_params=None,
                 op_ratios=None,
                 op_types=(tq.RX, tq.RY, tq.RZ, tq.CNOT),
                 seed=None,
                 qiskit_compatible=False,
                 ):
        super().__init__()
        self.n_ops = n_ops
        self.n_params = n_params
        assert n_params is not None or n_ops is not None
        self.wires = wires if isinstance(wires, Iterable) else [wires]
        self.n_wires = len(wires)

        op_types = op_types if isinstance(op_types, Iterable) else [op_types]
        if op_ratios is None:
            op_ratios = [1] * len(op_types)
        else:
            op_ratios = op_ratios if isinstance(op_ratios, Iterable) else [
                op_ratios]
        op_types_valid = []
        op_ratios_valid = []

        if qiskit_compatible:
            for op_type, op_ratio in zip(op_types, op_ratios):
                if op_type().name.lower() in QISKIT_INCOMPATIBLE_FUNC_NAMES:
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

    def rebuild_random_layer_from_op_list(self,
                                          n_ops_in,
                                          wires_in,
                                          op_list_in):
        """Used for loading random layer from checkpoint"""
        self.n_ops = n_ops_in
        self.wires = wires_in
        self.op_list = tq.QuantumModuleList()
        for op_in in op_list_in:
            op = tq.op_name_dict[op_in.name.lower()](
                has_params=op_in.has_params,
                trainable=op_in.trainable,
                wires=op_in.wires,
                n_wires=op_in.n_wires,
            )
            self.op_list.append(op)

    def build_random_layer(self):
        op_cnt = 0
        param_cnt = 0
        while True:
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
            op_cnt += 1
            param_cnt += op.num_params

            if self.n_ops is not None and op_cnt == self.n_ops:
                break
            elif self.n_ops is None and self.n_params is not None:
                if param_cnt == self.n_params:
                    break
                elif param_cnt > self.n_params:
                    """
                    the last operation has too many params and exceed the 
                    constraint, so need to remove it and sample another 
                    """
                    op_cnt -= 1
                    param_cnt -= op.num_params
                    del self.op_list[-1]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for op in self.op_list:
            op(q_device)


class RandomLayerAllTypes(RandomLayer):
    def __init__(self,
                 wires,
                 n_ops=None,
                 n_params=None,
                 op_ratios=None,
                 op_types=(tq.Hadamard,
                           tq.SHadamard,
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
                           tq.RZZ,
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
                 qiskit_compatible=False,
                 ):
        super().__init__(
            wires=wires,
            n_ops=n_ops,
            n_params=n_params,
            op_ratios=op_ratios,
            op_types=op_types,
            seed=seed,
            qiskit_compatible=qiskit_compatible,
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

class CXLayer(tq.QuantumModule):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires

    @tq.static_support
    def forward(self, q_dev):
        self.q_device = q_dev
        tqf.cnot(q_dev, wires=[0, 1], static=self.static_mode,
              parent_graph=self.graph)


class CXCXCXLayer(tq.QuantumModule):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires

    @tq.static_support
    def forward(self, q_dev):
        self.q_device = q_dev
        tqf.cnot(q_dev, wires=[0, 1], static=self.static_mode,
              parent_graph=self.graph)
        tqf.cnot(q_dev, wires=[1, 2], static=self.static_mode,
              parent_graph=self.graph)
        tqf.cnot(q_dev, wires=[2, 0], static=self.static_mode,
              parent_graph=self.graph)



class SWAPSWAPLayer(tq.QuantumModule):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires

    @tq.static_support
    def forward(self, q_dev):
        self.q_device = q_dev
        tqf.swap(q_dev, wires=[0, 1], static=self.static_mode,
              parent_graph=self.graph)
        tqf.swap(q_dev, wires=[1, 2], static=self.static_mode,
              parent_graph=self.graph)


class Op1QAllLayer(tq.QuantumModule):
    def __init__(self, op, n_wires: int, has_params=False, trainable=False):
        super().__init__()
        self.n_wires = n_wires
        self.op = op
        self.ops_all = tq.QuantumModuleList()
        for k in range(n_wires):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_wires):
            self.ops_all[k](q_device, wires=k)


class Op2QAllLayer(tq.QuantumModule):
    """pattern:
    circular = False
    jump = 1: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]
    jump = 2: [0, 2], [1, 3], [2, 4], [3, 5]
    jump = 3: [0, 3], [1, 4], [2, 5]
    jump = 4: [0, 4], [1, 5]
    jump = 5: [0, 5]

    circular = True
    jump = 1: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]
    jump = 2: [0, 2], [1, 3], [2, 4], [3, 5], [4, 0], [5, 1]
    jump = 3: [0, 3], [1, 4], [2, 5], [3, 0], [4, 1], [5, 2]
    jump = 4: [0, 4], [1, 5], [2, 0], [3, 1], [4, 2], [5, 3]
    jump = 5: [0, 5], [1, 0], [2, 1], [3, 2], [4, 3], [5, 4]
    """
    def __init__(self, op, n_wires: int, has_params=False, trainable=False,
                 wire_reverse=False, jump=1, circular=False):
        super().__init__()
        self.n_wires = n_wires
        self.jump = jump
        self.circular = circular
        self.op = op
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        if circular:
            n_ops = n_wires
        else:
            n_ops = n_wires - jump
        for k in range(n_ops):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(len(self.ops_all)):
            wires = [k, (k + self.jump) % self.n_wires]
            if self.wire_reverse:
                wires.reverse()
            self.ops_all[k](q_device, wires=wires)

class Op2QFit32Layer(tq.QuantumModule):
    def __init__(self, op, n_wires: int, has_params=False, trainable=False,
                 wire_reverse=False, jump=1, circular=False):
        super().__init__()
        self.n_wires = n_wires
        self.jump = jump
        self.circular = circular
        self.op = op
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        # if circular:
        #     n_ops = n_wires
        # else:
        #     n_ops = n_wires - jump
        n_ops = 32
        for k in range(n_ops):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(len(self.ops_all)):
            wires = [k % self.n_wires, (k + self.jump) % self.n_wires]
            if self.wire_reverse:
                wires.reverse()
            self.ops_all[k](q_device, wires=wires)


class Op2QButterflyLayer(tq.QuantumModule):
    """pattern: [0, 5], [1, 4], [2, 3]
    """
    def __init__(self, op, n_wires: int, has_params=False, trainable=False,
                 wire_reverse=False):
        super().__init__()
        self.n_wires = n_wires
        self.op = op
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        for k in range(n_wires // 2):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    def forward(self, q_device):
        for k in range(len(self.ops_all)):
            wires = [k, self.n_wires - 1 - k]
            if self.wire_reverse:
                wires.reverse()
            self.ops_all[k](q_device, wires=wires)


class Op2QDenseLayer(tq.QuantumModule):
    """pattern:
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]
    [1, 2], [1, 3], [1, 4], [1, 5]
    [2, 3], [2, 4], [2, 5]
    [3, 4], [3, 5]
    [4, 5]
    """
    def __init__(self, op, n_wires: int, has_params=False, trainable=False,
                 wire_reverse=False):
        super().__init__()
        self.n_wires = n_wires
        self.op = op
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        for k in range(self.n_wires * (self.n_wires - 1) // 2):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    def forward(self, q_device):
        k = 0
        for i in range(self.n_wires - 1):
            for j in range(i + 1, self.n_wires):
                wires = [i, j]
                if self.wire_reverse:
                    wires.reverse()
                self.ops_all[k](q_device, wires=wires)
                k += 1


class LayerTemplate0(tq.QuantumModule):
    def __init__(self, arch: dict = None):
        super().__init__()
        self.n_wires = arch['n_wires']
        self.arch = arch

        self.n_blocks = arch.get('n_blocks', None)
        self.n_layers_per_block = arch.get('n_layers_per_block', None)

        self.layers_all = self.build_layers()

    def build_layers(self):
        raise NotImplementedError

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for k in range(len(self.layers_all)):
            self.layers_all[k](q_device)


class U3CU3Layer0(LayerTemplate0):
    """u3 cu3 blocks"""
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.U3,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op2QAllLayer(
                    op=tq.CU3,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
        return layers_all


class CU3Layer0(LayerTemplate0):
    """u3 cu3 blocks"""
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op2QAllLayer(
                    op=tq.CU3,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=False))
        return layers_all


class CXRZSXLayer0(LayerTemplate0):
    """CXRZSX blocks"""
    def build_layers(self):
        layers_all = tq.QuantumModuleList()

        layers_all.append(
            Op1QAllLayer(
                op=tq.RZ,
                n_wires=self.n_wires,
                has_params=True,
                trainable=True))
        layers_all.append(
            Op2QAllLayer(
                op=tq.CNOT,
                n_wires=self.n_wires,
                jump=1,
                circular=False))
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op1QAllLayer(
                    op=tq.SX,
                    n_wires=self.n_wires,
                    has_params=False,
                    trainable=False))
        layers_all.append(
            Op1QAllLayer(
                op=tq.RZ,
                n_wires=self.n_wires,
                has_params=True,
                trainable=True))
        return layers_all


class SethLayer0(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op2QAllLayer(
                    op=tq.RZZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
        return layers_all


class SethLayer1(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op2QAllLayer(
                    op=tq.RZZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op2QAllLayer(
                    op=tq.RZZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
        return layers_all


class SethLayer2(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op2QFit32Layer(
                    op=tq.RZZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
        return layers_all


class RZZLayer0(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op2QAllLayer(
                    op=tq.RZZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
        return layers_all


class BarrenLayer0(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        layers_all.append(
            Op1QAllLayer(
                op=tq.SHadamard,
                n_wires=self.n_wires,)
        )
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RX,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op2QAllLayer(
                    op=tq.CZ,
                    n_wires=self.n_wires,
                    jump=1))
        return layers_all


class FarhiLayer0(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op2QAllLayer(
                    op=tq.RZX,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
            layers_all.append(
                Op2QAllLayer(
                    op=tq.RXX,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
        return layers_all


class MaxwellLayer0(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RX,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op1QAllLayer(
                    op=tq.S,
                    n_wires=self.n_wires))
            layers_all.append(
                Op2QAllLayer(
                    op=tq.CNOT,
                    n_wires=self.n_wires,
                    jump=1,
                    circular=True))

            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op1QAllLayer(
                    op=tq.T,
                    n_wires=self.n_wires))
            layers_all.append(
                Op2QAllLayer(
                    op=tq.SWAP,
                    n_wires=self.n_wires,
                    jump=1,
                    circular=True))

            layers_all.append(
                Op1QAllLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op1QAllLayer(
                    op=tq.Hadamard,
                    n_wires=self.n_wires))
            layers_all.append(
                Op2QAllLayer(
                    op=tq.SSWAP,
                    n_wires=self.n_wires,
                    jump=1,
                    circular=True))

            layers_all.append(
                Op1QAllLayer(
                    op=tq.U1,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True
                ))
            layers_all.append(
                Op2QAllLayer(
                    op=tq.CU3,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))

        return layers_all


class RYRYCXLayer0(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(CXLayer(n_wires=self.n_wires))
        return layers_all


class RYRYRYCXCXCXLayer0(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(CXCXCXLayer(n_wires=self.n_wires))
        return layers_all


class RYRYRYLayer0(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
        return layers_all


class RYRYRYSWAPSWAPLayer0(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(SWAPSWAPLayer(n_wires=self.n_wires))
        return layers_all


class SWAPSWAPLayer0(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(SWAPSWAPLayer(n_wires=self.n_wires))
        return layers_all


class RXYZCXLayer0(LayerTemplate0):
    def build_layers(self):
        layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RX,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op1QAllLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            layers_all.append(
                Op2QAllLayer(
                    op=tq.CNOT,
                    n_wires=self.n_wires,
                    jump=1,
                    circular=True))
        return layers_all


layer_name_dict = {
    'u3cu3_0': U3CU3Layer0,
    'cu3_0': CU3Layer0,
    'cxrzsx_0': CXRZSXLayer0,
    'seth_0': SethLayer0,
    'seth_1': SethLayer1,
    'seth_2': SethLayer2,
    'rzz_0': RZZLayer0,
    'barren_0': BarrenLayer0,
    'farhi_0': FarhiLayer0,
    'maxwell_0': MaxwellLayer0,
    'ryrycx': RYRYCXLayer0,
    'ryryrycxcxcx': RYRYRYCXCXCXLayer0,
    'ryryry': RYRYRYLayer0,
    'swapswap': SWAPSWAPLayer0, 
    'ryryryswapswap': RYRYRYSWAPSWAPLayer0,
    'rxyzcx_0': RXYZCXLayer0
}
