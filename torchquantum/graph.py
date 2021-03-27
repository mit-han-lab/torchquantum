import torch
import torchquantum as tq
import torchquantum.functional as tqf
import itertools
import functools
import numpy as np

from torchquantum.macro import C_DTYPE, ABC, ABC_ARRAY
from torchpack.utils.logging import logger


def encode_w(wires):
    return '.'.join(list(map(str, wires)))


def decode_w(w):
    return list(map(eval, w.split('.')))


def static_support(f):
    @functools.wraps(f)
    def forward_register_graph(*args, **kwargs):
        if args[0].static_mode and args[0].parent_graph is not None:
            args[0].parent_graph.add_op(args[0])
        res = f(*args, **kwargs)
        if args[0].static_mode and args[0].is_graph_top:
            # finish build graph, set flag
            args[0].set_graph_build_finish()
            args[0].static_forward(args[0].q_device)

        return res
    return forward_register_graph


class QuantumGraph(object):
    def __init__(self):
        self.is_list_finish = False
        self.module_list = []
        self.flat_module_list = []
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
        self.static_matrix_dict = {}

    def add_op(self, op: tq.Operator):
        if not self.is_list_finish:
            # graph construction is not finished, add the operation
            self.module_list.append(op)
        else:
            # graph construction is finished, no need to do anything.
            pass

    def add_func(self, name, wires, parent_graph, params=None, n_wires=None,
                 inverse=False):
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
            op.inverse = inverse

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
        self.build_flat_module_list(self.module_list)
        self.schedules = self.build_schedule_blockwise(wires_per_block)
        # self.schedules = self.build_schedule_layerwise()
        self.build_finish = True

    def forward(self, wires_per_block):
        if not self.build_finish:
            self.build(wires_per_block)

        self.build_static_matrix()
        self.apply_unitary()

    def build_static_matrix(self):
        # the optimization to speedup the static mode
        # 1. fixed unitary operators will share one matrix
        # 2. parameterized operators, the matrices for the same type are
        # computed together in one kernel call to speedup training
        matrix_params = {}

        # for wire_modules in self.wire_module_list:
        for module in self.flat_module_list:
            name = module.name
            if name in tq.Operator.fixed_ops:
                if name not in self.static_matrix_dict.keys():
                    # fixed operator, all share one static matrix
                    self.static_matrix_dict[module.name] = \
                        module.matrix.to(self.device)
            elif name in tq.Operator.parameterized_ops and name not in [
                'QubitUnitary',
                'QubitUnitaryFast',
                'TrainableUnitary',
                'TrainableUnitaryStrict',
                'MultiRZ',
            ]:
                # parameterized operators
                if name in matrix_params:
                    matrix_params[name].append(module.params)
                else:
                    matrix_params[name] = [module.params]
            elif name in [
                'QubitUnitary',
                'QubitUnitaryFast',
                'TrainableUnitary',
                'TrainableUnitaryStrict',
                'MultiRZ',
            ]:
                pass
            else:
                raise NotImplementedError(f"Module {name} not in list")

        ptrs = {}
        for name, param in matrix_params.items():
            param_cat = torch.cat(param, dim=0)
            self.static_matrix_dict[name] = tq.mat_dict[name.lower()](
                param_cat).to(self.device)
            if self.static_matrix_dict[name].dim() == 2:
                # in case there is only one matrix in this type of op
                self.static_matrix_dict[name] = self.static_matrix_dict[
                    name].unsqueeze(0)
            ptrs[name] = 0

        # for wire_modules in self.wire_module_list:
        for module in self.flat_module_list:
            name = module.name
            if name in tq.Operator.fixed_ops:
                module.static_matrix = self.static_matrix_dict[name]
            elif name in tq.Operator.parameterized_ops and name not in [
                'QubitUnitary',
                'QubitUnitaryFast',
                'TrainableUnitary',
                'TrainableUnitaryStrict',
                'MultiRZ'
            ]:
                shape0 = module.params.shape[0]
                module.static_matrix = self.static_matrix_dict[name][
                                       ptrs[name]: ptrs[name] + shape0]
                ptrs[name] += shape0
                if shape0 == 1:
                    module.static_matrix = module.static_matrix.squeeze(0)
            elif name in [
                'QubitUnitary',
                'QubitUnitaryFast',
                'TrainableUnitary',
                'TrainableUnitaryStrict',
            ]:
                module.static_matrix = module.params.squeeze(0)
            elif name in ['MultiRZ']:
                module.static_matrix = tq.mat_dict[name.lower()](
                    module.params, module.n_wires)
            else:
                raise NotImplementedError(f"Module {name} not in list")

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

    def build_flat_module_list(self, module_list=None):
        if module_list is None:
            module_list = self.module_list
        for module in module_list:
            if len(module.graph.module_list) == 0 and not \
                    isinstance(module, tq.Operator):
                logger.warning(f"Module with no operations exists!")
            if len(module.graph.module_list) == 0 and isinstance(module,
                                                                 tq.Operator):
                # leaf node
                self.flat_module_list.append(module)
            else:
                self.build_flat_module_list(module.graph.module_list)

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

        def is_front(wires, ptrs):
            return len(set([id(self.wire_module_list[w][ptrs[w]]) for w in
                            wires])) == 1

        def add_front_large_qubit_gate(ptrs, sches):
            # deal with the case when the number of wires in a block is
            # smaller than the number of wires in a gate
            for w in range(n_wires):
                if ptrs[w] == wire_module_len[w]:
                    continue

                module = self.wire_module_list[w][ptrs[w]]
                if len(module.wires) <= wires_per_block:
                    # number of wires in gate is smaller than or equal to
                    # that in a block
                    pass
                else:
                    # check whether the large gate are in the front
                    local_ws = [self.global2local_wire_mapping[wi]
                                for wi in module.wires]
                    if is_front(local_ws, ptrs):
                        sches.append({'wires': local_ws, 'modules': [module]})
                        for local_w in local_ws:
                            ptrs[local_w] += 1

        def comb_finish(combination, ptrs):
            return all(ptrs[w] == wire_module_len[w] for w in combination)

        def wires_in_comb(wires, combination):
            return all([w in combination for w in wires])

        schedules = []
        while not schedule_finish():
            add_front_large_qubit_gate(module_ptrs, schedules)

            # loop to get blocks
            # find the wire comb with max num of operations
            max_comb = None
            max_module_per_block = -1
            max_ptrs = None
            max_schedule = None
            for comb in itertools.combinations(list(range(n_wires)),
                                               min(wires_per_block, n_wires)):
                comb = list(comb)
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

    @staticmethod
    def acc_m_unitary_einsum(u, wires, module):
        if u.dim() % 2 == 1:
            is_u_batch = True
        else:
            is_u_batch = False

        device_wires = module.wires
        n_device_wires = len(device_wires)
        n_block_wires = len(wires)

        # module_matrix = module.matrix.to(self.device)
        module_matrix = module.static_matrix
        if module_matrix.dim() > 2:
            bsz = module_matrix.shape[0]
            is_module_matrix_batch = True
            shape_extension = [bsz]
        else:
            is_module_matrix_batch = False
            shape_extension = []

        if module.inverse:
            module_matrix = module_matrix.conj()
            if is_module_matrix_batch:
                module_matrix = module_matrix.permute(0, 2, 1)
            else:
                module_matrix = module_matrix.permute(1, 0)

        if n_device_wires > 1:
            module_matrix = module_matrix.view(shape_extension +
                                               [2] * n_device_wires * 2)

        # tensor indices for the quantum unitary
        n_block_letters = n_block_wires * 2
        unitary_indices = ABC[: n_block_letters]

        # indices of the quantum unitary affected by this operation
        locations_dim0 = [wires.index(wi) for wi in module.wires]
        affected_indices = "".join(ABC_ARRAY[locations_dim0].tolist())

        new_indices = ABC[n_block_letters: n_block_letters +
                          n_device_wires]
        try:
            assert n_block_letters + n_device_wires < 26
        except AssertionError:
            logger.exception(f"Einsum letters insufficient, please switch to "
                             f"bmm implementation.")
            raise AssertionError

        new_unitary_indices = functools.reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0],
                                                            idx_pair[1]),
            zip(affected_indices, new_indices),
            unitary_indices,
        )

        if is_u_batch:
            unitary_indices = ABC[-1] + unitary_indices

        if is_module_matrix_batch:
            new_indices = ABC[-1] + new_indices

        if is_u_batch or is_module_matrix_batch:
            new_unitary_indices = ABC[-1] + new_unitary_indices

        einsum_indices = f"{new_indices}{affected_indices}," \
                         f"{unitary_indices}->{new_unitary_indices}"

        new_unitary = torch.einsum(einsum_indices, module_matrix, u)

        return new_unitary

    @staticmethod
    def acc_m_unitary_bmm(u, global_wires, module):
        # compute the unitary of each block and apply to the device
        if u.dim() % 2 == 1:
            is_u_batch = True
        else:
            is_u_batch = False

        # module_matrix = module.matrix.to(self.device)
        module_matrix = module.static_matrix
        if module_matrix.dim() > 2:
            is_module_matrix_batch = True
        else:
            is_module_matrix_batch = False

        if module.inverse:
            module_matrix = module_matrix.conj()
            if is_module_matrix_batch:
                module_matrix = module_matrix.permute(0, 2, 1)
            else:
                module_matrix = module_matrix.permute(1, 0)

        n_global_wires = len(global_wires)
        m_global_wires = module.wires

        loc_all = list(range(n_global_wires * 2))
        loc_dim0 = [global_wires.index(wi) for wi in m_global_wires]
        loc_dim1 = [loc + n_global_wires for loc in loc_dim0]
        location = loc_dim0 + loc_dim1
        for loc in location:
            loc_all.remove(loc)
        permute_to = loc_all + location

        if is_u_batch:
            permute_to = [0] + [p + 1 for p in permute_to]
        permute_back = list(np.argsort(permute_to))

        original_shape = u.shape

        if is_u_batch:
            u = u.permute(permute_to).reshape([u.shape[0], -1, 2 ** len(
                m_global_wires), 2 ** len(m_global_wires)])
        else:
            u = u.permute(permute_to).reshape([-1, 2 ** len(m_global_wires),
                                               2 ** len(m_global_wires)])
            if u.dim() == 2:
                u = u.unsqueeze(0)

        if is_u_batch and is_module_matrix_batch:
            module_matrix = module_matrix.unsqueeze(-3)
        elif not is_u_batch and is_module_matrix_batch:
            bsz = module_matrix.shape[0]
            module_matrix = module_matrix.unsqueeze(-3)
            original_shape = [bsz] + list(original_shape)
            permute_back = [0] + [p + 1 for p in permute_back]
        # elif is_u_batch and not is_module_matrix_batch:
        #     pass
        # else:
        #     not is_u_batch and not is_module_matrix_batch:
        #     pass

        if not is_u_batch and not is_module_matrix_batch:
            new_u = module_matrix.expand(u.shape).bmm(u)
        else:
            new_u = module_matrix.matmul(u)
        new_u = new_u.view(original_shape).permute(permute_back)

        return new_u

    def get_unitary(self):
        """
        To get the whole unitary of the module, need to make sure all
        modules are in the same schedule
        """
        try:
            assert len(self.schedules) == 1
        except AssertionError:
            logger.exception(f"More than one block schedule in on module")

        return self.get_schedule_unitary(self.schedules[0])

    def get_schedule_unitary(self, schedule):
        # here some front large gates will need a larger unitary
        unitary = torch.eye(2 ** self.q_device.n_wires, dtype=C_DTYPE,
                            device=self.device).view(
            [2] * self.q_device.n_wires * 2)

        # global_wires = [self.local2global_wire_mapping[w] for w in comb]
        for m in schedule['modules']:
            unitary = self.acc_m_unitary_bmm(unitary, list(range(
                self.q_device.n_wires)), m)
        if unitary.dim() % 2 == 1:
            unitary = unitary.reshape(
                unitary.shape[0],
                2 ** self.q_device.n_wires,
                2 ** self.q_device.n_wires)
        else:
            unitary = unitary.reshape(2 ** self.q_device.n_wires,
                                      2 ** self.q_device.n_wires)
        return unitary

    def apply_unitary(self):
        for schedule in self.schedules:
            comb = schedule['wires']
            # here some front large gates will need a larger unitary
            unitary = torch.eye(2 ** len(comb), dtype=C_DTYPE,
                                device=self.device).view([2] *
                                                         len(comb) * 2)

            global_wires = [self.local2global_wire_mapping[w] for w in comb]
            for m in schedule['modules']:
                unitary = self.acc_m_unitary_bmm(unitary, global_wires, m)
            if unitary.dim() % 2 == 1:
                unitary = unitary.reshape(
                    unitary.shape[0],
                    2 ** len(comb),
                    2 ** len(comb))
            else:
                unitary = unitary.reshape(2 ** len(comb),
                                          2 ** len(comb))

            tqf.qubitunitaryfast(self.q_device, wires=global_wires,
                                 params=unitary)
