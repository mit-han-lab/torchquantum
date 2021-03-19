import torchquantum as tq
import itertools


__all__ = ['SuperQuantumModule', 'Super1QLayer', 'Super2QLayer']


class SuperQuantumModule(tq.QuantumModule):
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires

    def set_sample_config(self, sample_config):
        raise NotImplementedError

    @property
    def config_space(self):
        return None


class Super1QLayer(SuperQuantumModule):
    def __init__(self, op, n_wires: int, has_params=False, trainable=False):
        super().__init__(n_wires=n_wires)
        self.n_wires = n_wires
        self.op = op
        self.sample_wires = None
        self.ops_all = tq.QuantumModuleList()
        for k in range(n_wires):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    def set_sample_config(self, sample_config):
        self.sample_wires = sample_config['sample_wires'][0]

    def forward(self, q_device):
        for k in range(self.n_wires):
            if k in self.sample_wires:
                self.ops_all[k](q_device, wires=k)

    @property
    def config_space(self):
        return {'sample_wires': [list(range(self.n_wires))]}


class Super2QLayer(SuperQuantumModule):
    def __init__(self, op, n_wires: int, has_params=False, trainable=False,
                 wire_reverse=False):
        super().__init__(n_wires=n_wires)
        self.n_wires = n_wires
        self.op = op
        self.sample_wire_pairs = []
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse
        for k in range(n_wires):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    def set_sample_config(self, sample_config):
        self.sample_wire_pairs = sample_config['sample_wire_pairs'][0]

    def forward(self, q_device):
        for k in range(self.n_wires):
            if {k, (k + 1) % self.n_wires} in self.sample_wire_pairs:
                wires = sorted([k, (k + 1) % self.n_wires],
                               reverse=self.wire_reverse)
                self.ops_all[k](q_device, wires=wires)

    @property
    def config_space(self):
        return {'sample_wire_pairs': [list(map(set, itertools.combinations(
            list(range(4)), 2)))]}
