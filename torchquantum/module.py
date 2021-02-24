import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from abc import ABCMeta


__all__ = [
    'QuantumModule',
    'QuantumModuleList',
    'QuantumModuleDict'
]


class QuantumModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.static_mode = False
        self.graph = None
        self.parent_graph = None
        self.is_graph_top = False
        self.unitary = None
        self.wires = None
        self.n_wires = None
        self.q_device = None
        # this is for gpu or cpu, not q device
        self.device = None

    def static_on(self, is_graph_top=True):
        # register graph of itself and parent
        self.static_mode = True
        self.is_graph_top = is_graph_top
        if self.graph is None:
            self.graph = tq.QuantumGraph()

        for module in self.children():
            if isinstance(module, nn.ModuleList) or isinstance(module,
                                                               nn.ModuleDict):
                # if QuantumModuleList or QuantumModuleDict, its graph will
                # be the same as the parent graph because ModuleList and
                # ModuleDict do not call the forward function
                module.graph = self.graph
            module.parent_graph = self.graph
            if not isinstance(module, tq.QuantumDevice):
                module.static_on(is_graph_top=False)

    def static_off(self):
        self.static_mode = False
        self.graph = None
        for module in self.children():
            if not isinstance(module, tq.QuantumDevice):
                module.static_off()

    def set_graph_build_finish(self):
        self.graph.is_list_finish = True
        for module in self.graph.module_list:
            module.set_graph_build_finish()

    def static_forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        self.device = q_device.states.device
        self.graph.q_device = q_device
        self.graph.device = q_device.states.device
        # self.unitary, self.wires, self.n_wires = \
        self.graph.build_matrix()
        # tqf.qubitunitary(
        #     q_device=self.q_device,
        #     wires=self.wires,
        #     params=self.unitary
        # )


class QuantumModuleList(nn.ModuleList, QuantumModule, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QuantumModuleDict(nn.ModuleDict, QuantumModule, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def test():
    pass


if __name__ == '__main__':
    test()
