import torch.nn as nn
import torchquantum as tq
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

    def static_on(self):
        # register graph of itself and parent
        self.static_mode = True
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
            module.static_on()

    def static_off(self):
        for module in self.modules():
            if isinstance(module, tq.QuantumModule):
                module.static_mode = False


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
