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
