import numpy as np
import torchquantum as tq


class QuantumGraph(object):
    def __init__(self):
        self.graph = []

    def add_op(self, op: tq.Operator):
        self.graph.append(op)

    def add_func(self, name, wires, params=None, n_wires=None):
        op = tq.op_name_dict[name]()
        op.params = params
        op.n_wires = n_wires
        op.wires = wires

        self.graph.append(op)

    def flatten_graph(self):
        pass


