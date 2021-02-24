import torch
import torchquantum as tq
from torchquantum.macro import C_DTYPE


class QuantumGraph(object):
    def __init__(self):
        self.is_list_finish = False
        self.module_list = []
        self.wire_module_dict = {}
        self.wire_module_list = []
        self.local2global_wire_mapping = {}
        self.global2local_wire_mapping = {}
        self.schedules = []
        self.q_device = None
        # this is for gpu or cpu, not q device
        self.device = None

    def add_op(self, op: tq.Operator):
        if not self.is_list_finish:
            self.module_list.append(op)

    def add_func(self, name, wires, parent_graph, params=None, n_wires=None,):
        if not self.is_list_finish:
            op = tq.op_name_dict[name]()
            op.params = params
            op.n_wires = n_wires
            op.wires = wires
            op.graph = tq.QuantumGraph()
            op.parent_graph = parent_graph
            op.static_mode = True

            self.module_list.append(op)

    def flatten_graph(self):
        pass

    def build_matrix(self):
        self.build_wire_module_list(self.module_list)
        self.build_local_wire_module_list()
        self.build_schedule()
        self.build_unitary()

    def build_wire_module_list(self, module_list):
        for module in module_list:
            if len(module.graph.module_list) == 0:
                # leaf node
                for wire in module.wires:
                    if wire not in self.wire_module_dict.keys():
                        self.wire_module_dict[wire] = [module]
                    else:
                        self.wire_module_dict[wire].append(module)
            else:
                self.build_wire_module_list(module.graph.module_list)

    def build_local_wire_module_list(self):
        global_wires = sorted(list(self.wire_module_dict.keys()))
        local_wires = list(range(len(global_wires)))

        for k, global_wire in enumerate(global_wires):
            self.local2global_wire_mapping[local_wires[k]] = global_wire
            self.global2local_wire_mapping[global_wire] = local_wires[k]
            self.wire_module_list.append(self.wire_module_dict[global_wire])

    def build_schedule(self):
        # if multiple consecutive gate are applied to the same wires,
        # then merge them together in one scheduled layer
        n_wires = len(self.wire_module_list)
        module_ptrs = [0] * n_wires

        wire_module_len = [len(wire_module) for wire_module in
                           self.wire_module_list]

        def schedule_finish():
            return all([module_ptrs[k] == wire_module_len[k] for k in
                        range(len(module_ptrs))])

        def is_front(op, ptrs):
            ws = [self.global2local_wire_mapping[w] for w in op.wires]
            is_all_front = True
            for w in ws:
                if not id(op) == id(self.wire_module_list[w][ptrs[w]]):
                    is_all_front = False
                    break
            return is_all_front

        def update_ptrs(ws, ptrs):
            for w in ws:
                ptrs[w] += 1

        def has_conflict(wire_pairs, wires):
            if frozenset(wires) in wire_pairs:
                conflict = False
            else:
                occupied_wires = []
                for pair in wire_pairs:
                    occupied_wires.extend(list(pair))
                conflict = any([w in occupied_wires for w in wires])
            return conflict

        schedules = []
        while not schedule_finish():
            # loop to get layers of schedules
            schedule = {}
            for wire in range(n_wires):
                while module_ptrs[wire] < wire_module_len[wire]:
                    # loop to add multiple op to the same wire
                    current_op = self.wire_module_list[wire][module_ptrs[wire]]
                    op_wires = [self.global2local_wire_mapping[w] for w in
                                current_op.wires]
                    if is_front(current_op, module_ptrs) and \
                            not has_conflict(schedule.keys(), op_wires):
                        key = frozenset(op_wires)
                        if key in schedule:
                            schedule[key].append(current_op)
                        else:
                            schedule[key] = [current_op]
                        update_ptrs(op_wires, module_ptrs)
                    else:
                        break
            schedules.append(schedule)
        self.schedules = schedules

    def build_unitary(self):
        # iteratively compute unitary of one layer and compute the merged
        # unitary

        def same_wires_matmul(u_wire, m_op):
            return m_op.matmal(u_wire)

        for schedule in self.schedules:
            schedule_wire = []
            for wire_pair, ops in schedule.items():
                wire_list = list(wire_pair)
                schedule_wire.extend(wire_list)
                unitary_of_wire = None
                for op in ops:
                    if unitary_of_wire is None:
                        unitary_of_wire = op.matrix
                    else:
                        unitary_of_wire = op.matrix.matmul(unitary_of_wire)

