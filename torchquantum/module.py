import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = [
    'QuantumModule',
]


class QuantumModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.static_mode = False
        self.submodules = nn.ModuleList()
        self.tqf = None
        self.graph = None

    def static_on(self, top=True, graph=None):
        self.static_mode = True

        if top:
            self.graph = tq.QuantumGraph()
            self.tqf = self.graph
        else:
            self.tqf = graph

        for submodule in self.submodules:
            submodule.static_on(top=False, graph=self.graph)

    def static_off(self):
        self.static_mode = False
        self.tqf = tqf
        for submodule in self.submodules:
            submodule.static_off()


def test():
    pass


if __name__ == '__main__':
    test()
