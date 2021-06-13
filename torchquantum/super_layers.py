import torchquantum as tq
import itertools

from typing import List, Iterable

__all__ = ['SuperQuantumModule',
           'Super1QLayer',
           'Super2QLayer',
           'Super1QShareFrontLayer',
           'Super1QSingleWireLayer',
           'Super1QAllButOneLayer',
           'Super2QAllShareFrontLayer',
           'Super2QAllLayer',
           'Super2QAlterLayer',
           'super_layer_name_dict',
           ]


def get_combs(inset: List, n=None) -> List[List]:
    all_combs = []
    if n is None:
        # all possible combinations, with different #elements in a set
        for k in range(1, len(inset) + 1):
            all_combs.extend(list(map(list, itertools.combinations(inset, k))))
    elif isinstance(n, int):
        all_combs.extend(list(map(list, itertools.combinations(inset, n))))
    elif isinstance(n, Iterable):
        for k in n:
            all_combs.extend(list(map(list, itertools.combinations(inset, k))))

    return all_combs


class SuperQuantumModule(tq.QuantumModule):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires
        self.sample_arch = None

    def set_sample_arch(self, sample_arch):
        self.sample_arch = sample_arch

    @property
    def arch_space(self):
        return None

    def count_sample_params(self):
        raise NotImplementedError


class Super1QLayer(SuperQuantumModule):
    def __init__(self, op, n_wires: int, has_params=False, trainable=False):
        super().__init__(n_wires=n_wires)
        self.op = op
        self.sample_wires = None
        self.ops_all = tq.QuantumModuleList()
        for k in range(n_wires):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_wires):
            if k in self.sample_arch:
                self.ops_all[k](q_device, wires=k)

    @property
    def arch_space(self):
        choices = list(range(self.n_wires))
        return get_combs(choices)

    def count_sample_params(self):
        return len(self.sample_arch) * self.op.num_params


class Super2QLayer(SuperQuantumModule):
    def __init__(self, op, n_wires: int, has_params=False, trainable=False,
                 wire_reverse=False):
        super().__init__(n_wires=n_wires)
        self.op = op
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse
        for k in range(n_wires):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_wires):
            if [k, (k + 1) % self.n_wires] in self.sample_arch or \
                    [(k + 1) % self.n_wires, k] in self.sample_arch:
                wires = sorted([k, (k + 1) % self.n_wires],
                               reverse=self.wire_reverse)
                self.ops_all[k](q_device, wires=wires)

    @property
    def arch_space(self):
        choices = list(map(list, itertools.combinations(list(range(
            self.n_wires)), 2)))
        return get_combs(choices)

    def count_sample_params(self):
        return len(self.sample_arch) * self.op.num_params


class Super1QShareFrontLayer(SuperQuantumModule):
    """Share the front wires, the rest can be added"""
    def __init__(self,
                 op,
                 n_wires: int,
                 n_front_share_wires: int,
                 has_params=False,
                 trainable=False,):
        super().__init__(n_wires=n_wires)
        self.n_front_share_wires = n_front_share_wires
        self.op = op
        self.ops_all = tq.QuantumModuleList()
        for k in range(n_wires):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        self.q_device = q_device
        for k in range(self.n_wires):
            if k < self.sample_arch:
                self.ops_all[k](q_device, wires=k)

    @property
    def arch_space(self):
        return list(range(self.n_front_share_wires, self.n_wires + 1))

    def count_sample_params(self):
        return min(self.sample_arch, len(self.ops_all)) * self.op.num_params


class Super1QSingleWireLayer(SuperQuantumModule):
    """Only one wire will have a gate"""
    def __init__(self,
                 op,
                 n_wires: int,
                 has_params=False,
                 trainable=False,):
        super().__init__(n_wires=n_wires)
        self.op = op
        self.ops_all = tq.QuantumModuleList()
        for k in range(n_wires):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_wires):
            if k == self.sample_arch:
                self.ops_all[k](q_device, wires=k)

    @property
    def arch_space(self):
        return list(range(self.n_wires))

    def count_sample_params(self):
        return self.op.num_params


class Super1QAllButOneLayer(SuperQuantumModule):
    """Only one wire will NOT have the gate"""
    def __init__(self,
                 op,
                 n_wires: int,
                 has_params=False,
                 trainable=False,):
        super().__init__(n_wires=n_wires)
        self.op = op
        self.ops_all = tq.QuantumModuleList()
        for k in range(n_wires):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_wires):
            if k != self.sample_arch:
                self.ops_all[k](q_device, wires=k)

    @property
    def arch_space(self):
        return list(range(self.n_wires))

    def count_sample_params(self):
        return (self.n_wires - 1) * self.op.num_params


class Super2QAllShareFrontLayer(SuperQuantumModule):
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
    def __init__(self, op, n_wires: int, n_front_share_ops: int,
                 has_params=False, trainable=False,
                 wire_reverse=False, jump=1, circular=False):
        super().__init__(n_wires=n_wires)
        self.op = op
        self.n_front_share_ops = n_front_share_ops
        self.jump = jump
        self.circular = circular

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        self.ops_all = tq.QuantumModuleList()
        if circular:
            self.n_ops = n_wires
        else:
            self.n_ops = n_wires - jump
        for k in range(self.n_ops):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_ops):
            if k < self.sample_arch:
                wires = [k, (k + self.jump) % self.n_wires]
                if self.wire_reverse:
                    wires.reverse()
                self.ops_all[k](q_device, wires=wires)

    @property
    def arch_space(self):
        choices = list(range(self.n_front_share_ops, self.n_ops + 1))
        return choices

    def count_sample_params(self):
        return min(self.sample_arch, self.n_ops) * self.op.num_params


class Super2QAllLayer(SuperQuantumModule):
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
    def __init__(self, op, n_wires: int,
                 has_params=False, trainable=False,
                 wire_reverse=False, jump=1, circular=False):
        super().__init__(n_wires=n_wires)
        self.op = op
        self.jump = jump
        self.circular = circular

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        self.ops_all = tq.QuantumModuleList()
        if circular:
            self.n_ops = n_wires
        else:
            self.n_ops = n_wires - jump
        for k in range(self.n_ops):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_ops):
            wires = [k, (k + self.jump) % self.n_wires]
            if self.wire_reverse:
                wires.reverse()

            if wires in self.sample_arch or list(reversed(wires)) in \
                    self.sample_arch:
                self.ops_all[k](q_device, wires=wires)

    @property
    def arch_space(self):
        choices = []
        for k in range(self.n_ops):
            wires = [k, (k + self.jump) % self.n_wires]
            choices.append(wires)
        return get_combs(choices)

    def count_sample_params(self):
        return len(self.sample_arch) * self.op.num_params


class Super2QAlterLayer(SuperQuantumModule):
    """pattern
    jump = 1: [0, 1], [2, 3], [4, 5], [1, 2], [3, 4], [5, 6]
    jump = 2: [0, 2], [4, 6], [2, 4]
    jump = 3: [0, 3], [3, 6]
    jump = 4: [0, 4]
    jump = 5: [0, 5]
    jump = 6: [0, 6]
    """
    def __init__(self, op, n_wires: int,
                 has_params=False, trainable=False,
                 wire_reverse=False, jump=1):
        super().__init__(n_wires=n_wires)
        self.op = op
        self.jump = jump

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        self.ops_all = tq.QuantumModuleList()
        self.n_ops = (n_wires - 1) // jump
        for k in range(self.n_ops):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))
        self.wires_choices = []
        k = 0
        while k < n_wires:
            if k < n_wires and k + jump < n_wires:
                self.wires_choices.append([k, k + jump])
            k += jump * 2

        k = jump
        while k < n_wires:
            if k < n_wires and k + jump < n_wires:
                self.wires_choices.append([k, k + jump])
            k += jump * 2

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_ops):
            wires = self.wires_choices[k]
            if self.wire_reverse:
                wires.reverse()

            if wires in self.sample_arch or list(reversed(wires)) in \
                    self.sample_arch:
                self.ops_all[k](q_device, wires=wires)

    @property
    def arch_space(self):
        return get_combs(self.wires_choices)

    def count_sample_params(self):
        return len(self.sample_arch) * self.op.num_params


class Super2QAlterShareFrontLayer(SuperQuantumModule):
    """pattern
    jump = 1: [0, 1], [2, 3], [4, 5], [1, 2], [3, 4], [5, 6]
    jump = 2: [0, 2], [4, 6], [2, 4]
    jump = 3: [0, 3], [3, 6]
    jump = 4: [0, 4]
    jump = 5: [0, 5]
    jump = 6: [0, 6]
    """
    def __init__(self, op, n_wires: int, n_front_share_ops: int,
                 has_params=False, trainable=False,
                 wire_reverse=False, jump=1):
        super().__init__(n_wires=n_wires)
        self.op = op
        self.n_front_share_ops = n_front_share_ops
        self.jump = jump

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        self.ops_all = tq.QuantumModuleList()
        self.n_ops = (n_wires - 1) // jump
        for k in range(self.n_ops):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))
        self.wires_choices = []
        k = 0
        while k < n_wires:
            if k < n_wires and k + jump < n_wires:
                self.wires_choices.append([k, k + jump])
            k += jump * 2

        k = jump
        while k < n_wires:
            if k < n_wires and k + jump < n_wires:
                self.wires_choices.append([k, k + jump])
            k += jump * 2

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_ops):
            if k < self.sample_arch:
                wires = self.wires_choices[k]
                if self.wire_reverse:
                    wires.reverse()

                self.ops_all[k](q_device, wires=wires)

    @property
    def arch_space(self):
        choices = list(range(self.n_front_share_ops, self.n_ops + 1))
        return choices

    def count_sample_params(self):
        return min(self.sample_arch, self.n_ops) * self.op.num_params


class SuperLayerTemplate0(SuperQuantumModule):
    def __init__(self, arch: dict = None):
        super().__init__(n_wires=arch['n_wires'])
        self.arch = arch

        self.n_front_share_wires = arch.get('n_front_share_wires', None)
        self.n_front_share_ops = arch.get('n_front_share_ops', None)

        self.n_blocks = arch.get('n_blocks', None)
        self.n_layers_per_block = arch.get('n_layers_per_block', None)
        self.n_front_share_blocks = arch.get('n_front_share_blocks', None)

        self.sample_n_blocks = None

        self.super_layers_all = self.build_super_layers()

    def build_super_layers(self):
        raise NotImplementedError

    def set_sample_arch(self, sample_arch):
        for k, layer_arch in enumerate(sample_arch[:-1]):
            self.super_layers_all[k].set_sample_arch(layer_arch)
        self.sample_n_blocks = sample_arch[-1]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for k in range(len(self.super_layers_all)):
            if k < self.sample_n_blocks * self.n_layers_per_block:
                self.super_layers_all[k](q_device)

    def count_sample_params(self):
        n_params = 0
        for layer_idx, layer in enumerate(self.super_layers_all):
            if layer_idx < self.sample_n_blocks * self.n_layers_per_block:
                n_params += layer.count_sample_params()
        return n_params


class SuperU3CU3ShareFrontLayer0(SuperLayerTemplate0):
    """u3 cu3 blocks"""
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                Super1QShareFrontLayer(
                    op=tq.U3,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super2QAllShareFrontLayer(
                    op=tq.CU3,
                    n_wires=self.n_wires,
                    n_front_share_ops=self.n_front_share_ops,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
        return super_layers_all


class SuperU3CU3ArbitraryLayer0(SuperLayerTemplate0):
    """u3 cu3 blocks arbitrary n gates"""
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                Super1QLayer(
                    op=tq.U3,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super2QAllLayer(
                    op=tq.CU3,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
        return super_layers_all


class SuperIBMBasisShareFrontLayer0(SuperLayerTemplate0):
    """cnot rz sx x share front blocks"""
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                Super1QShareFrontLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super1QShareFrontLayer(
                    op=tq.PauliX,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires))
            super_layers_all.append(
                Super1QShareFrontLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super1QShareFrontLayer(
                    op=tq.SX,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires))
            super_layers_all.append(
                Super1QShareFrontLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super2QAllShareFrontLayer(
                    op=tq.CNOT,
                    n_wires=self.n_wires,
                    n_front_share_ops=self.n_front_share_ops,
                    jump=1,
                    circular=True))
        return super_layers_all


class SuperIBMBasisArbitraryLayer0(SuperLayerTemplate0):
    """cnot rz sx x blocks arbitrary n gates"""
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                Super1QLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super1QLayer(
                    op=tq.PauliX,
                    n_wires=self.n_wires))
            super_layers_all.append(
                Super1QLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super1QLayer(
                    op=tq.SX,
                    n_wires=self.n_wires))
            super_layers_all.append(
                Super1QLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super2QAllLayer(
                    op=tq.CNOT,
                    n_wires=self.n_wires,
                    jump=1,
                    circular=True))
        return super_layers_all


class SuperSethArbitraryLayer0(SuperLayerTemplate0):
    """
    zz and ry blocks arbitrary n gates, from Seth Lloyd paper
    https://arxiv.org/pdf/2001.03622.pdf
    """
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                Super2QAllLayer(
                    op=tq.RZZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
            super_layers_all.append(
                Super1QLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
        return super_layers_all


class SuperSethShareFrontLayer0(SuperLayerTemplate0):
    """
    zz and ry blocks share front n gates, from Seth Lloyd paper
    https://arxiv.org/pdf/2001.03622.pdf
    """
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()
        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                Super2QAllShareFrontLayer(
                    op=tq.RZZ,
                    n_wires=self.n_wires,
                    n_front_share_ops=self.n_front_share_ops,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
            super_layers_all.append(
                Super1QShareFrontLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
        return super_layers_all


class SuperBarrenArbitraryLayer0(SuperLayerTemplate0):
    """
    rx ry rz and cz blocks arbitrary n gates, from Barren plateaus paper
    https://arxiv.org/pdf/1803.11173.pdf
    """
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()

        super_layers_all.append(
            Super1QLayer(op=tq.SHadamard, n_wires=self.n_wires))

        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                Super1QLayer(
                    op=tq.RX,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super1QLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super1QLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super2QAlterLayer(
                    op=tq.CZ,
                    n_wires=self.n_wires,
                    jump=1))
        return super_layers_all


class SuperBarrenShareFrontLayer0(SuperLayerTemplate0):
    """
    rx ry rz and cz blocks share front n gates, from Barren plateaus paper
    https://arxiv.org/pdf/1803.11173.pdf
    """
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()

        super_layers_all.append(
            Super1QShareFrontLayer(
                op=tq.SHadamard,
                n_wires=self.n_wires,
                n_front_share_wires=self.n_front_share_wires
            ))

        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                Super1QShareFrontLayer(
                    op=tq.RX,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super1QShareFrontLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super1QShareFrontLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                Super2QAlterShareFrontLayer(
                    op=tq.CZ,
                    n_wires=self.n_wires,
                    n_front_share_ops=self.n_front_share_ops,
                    jump=1))
        return super_layers_all


class SuperFarhiArbitraryLayer0(SuperLayerTemplate0):
    """
    zx and xx blocks arbitrary n gates, from Farhi paper
    https://arxiv.org/pdf/1802.06002.pdf
    """
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()

        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                tq.Super2QAllLayer(
                    op=tq.RZX,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
            super_layers_all.append(
                tq.Super2QAllLayer(
                    op=tq.RXX,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
        return super_layers_all


class SuperFarhiShareFrontLayer0(SuperLayerTemplate0):
    """
    zx and xx blocks share front n gates, from Farhi paper
    https://arxiv.org/pdf/1802.06002.pdf
    """
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()

        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                tq.Super2QAllShareFrontLayer(
                    op=tq.RZX,
                    n_wires=self.n_wires,
                    n_front_share_ops=self.n_front_share_ops,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
            super_layers_all.append(
                tq.Super2QAllShareFrontLayer(
                    op=tq.RXX,
                    n_wires=self.n_wires,
                    n_front_share_ops=self.n_front_share_ops,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))
        return super_layers_all


class SuperMaxwellArbitraryLayer0(SuperLayerTemplate0):
    """
    rx, s, cnot, ry, t, swap, rz, h, sswap, u1, cu3,
    blocks arbitrary n gates, from Maxwell paper
    https://arxiv.org/pdf/1904.04767.pdf
    """
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()

        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                tq.Super1QLayer(
                    op=tq.RX,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                tq.Super1QLayer(
                    op=tq.S,
                    n_wires=self.n_wires))
            super_layers_all.append(
                tq.Super2QAllLayer(
                    op=tq.CNOT,
                    n_wires=self.n_wires,
                    jump=1,
                    circular=True))

            super_layers_all.append(
                tq.Super1QLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                tq.Super1QLayer(
                    op=tq.T,
                    n_wires=self.n_wires))
            super_layers_all.append(
                tq.Super2QAllLayer(
                    op=tq.SWAP,
                    n_wires=self.n_wires,
                    jump=1,
                    circular=True))

            super_layers_all.append(
                tq.Super1QLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                tq.Super1QLayer(
                    op=tq.T,
                    n_wires=self.n_wires))
            super_layers_all.append(
                tq.Super2QAllLayer(
                    op=tq.SSWAP,
                    n_wires=self.n_wires,
                    jump=1,
                    circular=True))

            super_layers_all.append(
                tq.Super1QLayer(
                    op=tq.U1,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                tq.Super2QAllLayer(
                    op=tq.CU3,
                    n_wires=self.n_wires,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))

        return super_layers_all


class SuperMaxwellShareFrontLayer0(SuperLayerTemplate0):
    """
    rx, s, cnot, ry, t, swap, rz, h, sswap, u1, cu3,
    blocks share front n gates, from Maxwell paper
    https://arxiv.org/pdf/1904.04767.pdf
    """
    def build_super_layers(self):
        super_layers_all = tq.QuantumModuleList()

        for k in range(self.arch['n_blocks']):
            super_layers_all.append(
                tq.Super1QShareFrontLayer(
                    op=tq.RX,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                tq.Super1QShareFrontLayer(
                    op=tq.S,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires))
            super_layers_all.append(
                tq.Super2QAllShareFrontLayer(
                    op=tq.CNOT,
                    n_wires=self.n_wires,
                    n_front_share_ops=self.n_front_share_ops,
                    jump=1,
                    circular=True))

            super_layers_all.append(
                tq.Super1QShareFrontLayer(
                    op=tq.RY,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                tq.Super1QShareFrontLayer(
                    op=tq.T,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires))
            super_layers_all.append(
                tq.Super2QAllShareFrontLayer(
                    op=tq.SWAP,
                    n_wires=self.n_wires,
                    n_front_share_ops=self.n_front_share_ops,
                    jump=1,
                    circular=True))

            super_layers_all.append(
                tq.Super1QShareFrontLayer(
                    op=tq.RZ,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                tq.Super1QShareFrontLayer(
                    op=tq.T,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires))
            super_layers_all.append(
                tq.Super2QAllShareFrontLayer(
                    op=tq.SSWAP,
                    n_wires=self.n_wires,
                    n_front_share_ops=self.n_front_share_ops,
                    jump=1,
                    circular=True))

            super_layers_all.append(
                tq.Super1QShareFrontLayer(
                    op=tq.U1,
                    n_wires=self.n_wires,
                    n_front_share_wires=self.n_front_share_wires,
                    has_params=True,
                    trainable=True))
            super_layers_all.append(
                tq.Super2QAllShareFrontLayer(
                    op=tq.CU3,
                    n_wires=self.n_wires,
                    n_front_share_ops=self.n_front_share_ops,
                    has_params=True,
                    trainable=True,
                    jump=1,
                    circular=True))

        return super_layers_all


super_layer_name_dict = {
    'u3cu3_s0': SuperU3CU3ShareFrontLayer0,
    'seth_s0': SuperSethShareFrontLayer0,
    'barren_s0': SuperBarrenShareFrontLayer0,
    'farhi_s0': SuperFarhiShareFrontLayer0,
    'maxwell_s0': SuperMaxwellShareFrontLayer0,
    'ibmbasis_s0': SuperIBMBasisShareFrontLayer0,
    'u3cu3_a0': SuperU3CU3ArbitraryLayer0,
    'seth_a0': SuperSethArbitraryLayer0,
    'barren_a0': SuperBarrenArbitraryLayer0,
    'farhi_a0': SuperFarhiArbitraryLayer0,
    'maxwell_a0': SuperMaxwellArbitraryLayer0,
    'ibmbasis_a0': SuperIBMBasisArbitraryLayer0,
}
