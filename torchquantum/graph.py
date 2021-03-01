import torch
import torchquantum as tq
import torchquantum.functional as tqf
import itertools
import numpy as np

from torchquantum.macro import C_DTYPE


def encode_w(wires):
    return '.'.join(list(map(str, wires)))


def decode_w(w):
    return list(map(eval, w.split('.')))


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
        self.func_list = []
        self.func_ptr = 0
        self.num_func = 0
        self.build_finish = False

    def add_op(self, op: tq.Operator):
        if not self.is_list_finish:
            # graph construction is not finished, add the operation
            self.module_list.append(op)
        else:
            # graph construction is finished, no need to do anything.
            pass

    def add_func(self, name, wires, parent_graph, params=None, n_wires=None,):
        if not self.is_list_finish:
            # graph construction is not finished, build a new operation and
            # add the operation to the graph
            op = tq.op_name_dict[name]()
            op.params = params
            op.n_wires = n_wires
            op.wires = wires
            op.graph = tq.QuantumGraph()
            op.parent_graph = parent_graph
            op.static_mode = True

            self.module_list.append(op)
            self.func_list.append(op)
            self.num_func += 1
        else:
            # graph construction is finished, update the parameters of the
            # new operation
            op = self.func_list[self.func_ptr]
            op.params = params

            self.func_ptr = (self.func_ptr + 1) % self.num_func

    def build(self, wires_per_block):
        self.build_wire_module_dict(self.module_list)
        self.build_local_wire_module_list()
        self.schedules = self.build_schedule_blockwise(wires_per_block)
        # self.schedules = self.build_schedule_layerwise()
        self.build_finish = True

    def forward(self, wires_per_block):
        if not self.build_finish:
            self.build(wires_per_block)

        self.apply_unitary(wires_per_block)

    def build_wire_module_dict(self, module_list):
        for module in module_list:
            if len(module.graph.module_list) == 0:
                # leaf node
                for wire in module.wires:
                    if wire not in self.wire_module_dict.keys():
                        self.wire_module_dict[wire] = [module]
                    else:
                        self.wire_module_dict[wire].append(module)
            else:
                self.build_wire_module_dict(module.graph.module_list)

    def build_local_wire_module_list(self):
        global_wires = sorted(list(self.wire_module_dict.keys()))
        local_wires = list(range(len(global_wires)))

        for k, global_wire in enumerate(global_wires):
            self.local2global_wire_mapping[local_wires[k]] = global_wire
            self.global2local_wire_mapping[global_wire] = local_wires[k]
            self.wire_module_list.append(self.wire_module_dict[global_wire])

    def build_schedule_blockwise(self, wires_per_block):
        # greedily search for the block containing most number of gates per
        # block. The block contains wires_per_block wires as a hyper-parameter
        n_wires = len(self.wire_module_list)
        module_ptrs = [0] * n_wires
        wire_module_len = [len(wire_module) for wire_module in
                           self.wire_module_list]

        def schedule_finish():
            return all([module_ptrs[k] == wire_module_len[k] for k in
                        range(len(module_ptrs))])

        def check_front_multi_qubit_gate():
            return None

        def comb_finish(combination, ptrs):
            return all(ptrs[w] == wire_module_len[w] for w in combination)

        def is_front(wires, ptrs):
            return len(set([id(self.wire_module_list[w][ptrs[w]]) for w in
                       wires])) == 1

        def wires_in_comb(wires, combination):
            return all([w in combination for w in wires])

        schedules = []
        while not schedule_finish():
            check_front_multi_qubit_gate()
            # loop to get blocks
            # find the wire comb with max num of operations
            max_comb = None
            max_module_per_block = -1
            max_ptrs = None
            max_schedule = None
            for comb in itertools.combinations(list(range(n_wires)),
                                               wires_per_block):
                comb_module_per_block = 0
                comb_ptrs = module_ptrs.copy()
                comb_schedule = []
                # check how many gates can be added to the block
                while not comb_finish(comb, comb_ptrs):
                    has_changes = False
                    for wire in comb:
                        if comb_ptrs[wire] == wire_module_len[wire]:
                            continue
                        m = self.wire_module_list[wire][comb_ptrs[wire]]
                        module_wires = [
                            self.global2local_wire_mapping[w] for w in m.wires]
                        if is_front(module_wires, comb_ptrs) and \
                                wires_in_comb(module_wires, comb):
                            comb_module_per_block += 1
                            for w in module_wires:
                                comb_ptrs[w] += 1
                            comb_schedule.append(m)
                            has_changes = True
                        else:
                            continue
                    if not has_changes:
                        break

                if comb_module_per_block > max_module_per_block:
                    max_comb = comb
                    max_module_per_block = comb_module_per_block
                    max_ptrs = comb_ptrs
                    max_schedule = comb_schedule

            schedules.append({'wires': max_comb, 'modules': max_schedule})
            module_ptrs = max_ptrs.copy()

        return schedules

    def build_schedule_layerwise(self):
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
            if encode_w(wires) in wire_pairs:
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
                        key = encode_w(op_wires)
                        if key in schedule:
                            schedule[key].append(current_op)
                        else:
                            schedule[key] = [current_op]
                        update_ptrs(op_wires, module_ptrs)
                    else:
                        break
            schedules.append(schedule)
        return schedules

    def apply_unitary(self, wires_per_block):
        # compute the unitary of each block and apply to the device
        def acc_m_unitary(u, wires, module):
            u2w_mapping = {}
            w2u_mapping = {}
            for k, w in enumerate(wires):
                # mapping between the block unitary wire (u) and gate local
                # wire (w)
                w2u_mapping[w] = k
                u2w_mapping[k] = w

            if u.dim() % 2 == 1:
                is_u_batch = True
            else:
                is_u_batch = False

            module_matrix = module.matrix.to(self.device)
            if module_matrix.dim() > 2:
                is_module_matrix_batch = True
            else:
                is_module_matrix_batch = False

            m_wires = [self.global2local_wire_mapping[w] for w in module.wires]
            m_first_dim = []
            m_second_dim = []
            remain_first_dim = []
            remain_second_dim = []

            for k in m_wires:
                m_first_dim.append(w2u_mapping[k])
                m_second_dim.append(w2u_mapping[k] + len(wires))

            for k in range(len(wires)):
                if u2w_mapping[k] not in m_wires:
                    remain_first_dim.append(k)
                    remain_second_dim.append(k + len(wires))

            permute_to = remain_first_dim + remain_second_dim + m_first_dim \
                + m_second_dim
            if is_u_batch:
                permute_to = [0] + [p + 1 for p in permute_to]
            permute_back = list(np.argsort(permute_to))

            original_shape = u.shape

            if is_u_batch:
                u = u.permute(permute_to).reshape([u.shape[0], -1, 2 ** len(
                    m_wires), 2 ** len(m_wires)])
            else:
                u = u.permute(permute_to).reshape([-1, 2 ** len(m_wires),
                                                   2 ** len(m_wires)])

                if u.dim() == 2:
                    u = u.unsqueeze(0)

            if is_u_batch and is_module_matrix_batch:
                module_matrix = module_matrix.unsqueeze(-3)
            elif not is_u_batch and is_module_matrix_batch:
                bsz = module_matrix.shape[0]
                module_matrix = module_matrix.unsqueeze(-3)
                original_shape = [bsz] + list(original_shape)
                permute_back = [0] + [p + 1 for p in permute_back]
            elif is_u_batch and not is_module_matrix_batch:
                pass
            else:
                # not is_u_batch and not is_module_matrix_batch:
                pass

            new_u = module_matrix.matmul(u).view(original_shape).permute(
                permute_back)

            return new_u

        for schedule in self.schedules:
            comb = schedule['wires']
            unitary = torch.eye(2 ** wires_per_block, dtype=C_DTYPE,
                                device=self.device).view([2] *
                                                         wires_per_block * 2)
            for m in schedule['modules']:
                unitary = acc_m_unitary(unitary, comb, m)
            if unitary.dim() % 2 == 1:
                unitary = unitary.reshape(
                    unitary.shape[0],
                    2 ** wires_per_block,
                    2 ** wires_per_block)
            else:
                unitary = unitary.reshape(2 ** wires_per_block,
                                          2 ** wires_per_block)
            global_wires = [self.local2global_wire_mapping[w] for w in comb]
            tqf.qubitunitary(self.q_device, wires=global_wires, params=unitary)
