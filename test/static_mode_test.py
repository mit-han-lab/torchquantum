# import torch
# import torchquantum as tq
#
# from torchquantum.encoding import encoder_op_list_name_dict
# from torchquantum.layers import layer_name_dict
# import numpy as np
# import datetime
#
#
# __all__ = ['QuantumNode',
#            'build_nodes',
#            ]
#
#
# class QuantumNode(tq.QuantumModule):
#     """
#     a quantum node contains a q device, encoder, q layer and measure
#     """
#     def __init__(self, arch, act_norm, node_id):
#         super().__init__()
#         self.arch = arch
#         self.q_device = tq.QuantumDevice(n_wires=arch['n_wires'])
#         self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict[
#                                              arch['encoder_op_list_name']])
#         self.q_layer = layer_name_dict[arch['q_layer_name']](arch)
#         self.measure = tq.MeasureAll(tq.PauliZ)
#         self.act_norm = act_norm
#         self.x_before_add_noise = None
#         self.x_before_add_noise_second = None
#         self.x_before_act_quant = None
#         self.x_before_norm = None
#         self.circuit_in = None
#         self.circuit_out = None
#         self.shift_this_step = np.array([True] * len(list(self.q_layer.parameters())))
#         self.cool_down = [0] * len(list(self.q_layer.parameters()))
#         self.triger_cd = [0] * len(list(self.q_layer.parameters()))
#         if self.act_norm == 'batch_norm' or \
#                 self.act_norm == 'batch_norm_no_last':
#             self.bn = torch.nn.BatchNorm1d(
#                 num_features=arch['n_wires'],
#                 momentum=None,
#                 affine=False,
#                 track_running_stats=False
#             )
#         self.node_id = node_id
#         self.pre_specified_mean_std = None
#         self.grad_qlayer = None
#         self.grad_encoder = None
#
#     def forward(self, x, use_qiskit=False, is_last_node=False, parallel=True):
#         if use_qiskit:
#             x = self.qiskit_processor.process_parameterized(
#                 self.q_device,
#                 self.encoder,
#                 self.q_layer,
#                 self.measure,
#                 x,
#                 parallel=parallel
#             )
#         else:
#             self.encoder(self.q_device, x)
#             self.q_layer(self.q_device)
#             x = self.measure(self.q_device)
#
#         self.x_before_add_noise = x.clone()
#
#         if isinstance(self.noise_model_tq, tq.NoiseModelTQActivation):
#             x = self.noise_model_tq.add_noise(x, self.node_id,
#                                               is_after_norm=False)
#
#         self.x_before_norm = x.clone()
#
#         if self.act_norm == 'layer_norm':
#             x = (x - x.mean(-1).unsqueeze(-1)) / x.std(-1).unsqueeze(-1)
#         elif self.act_norm == 'batch_norm':
#             if self.pre_specified_mean_std is None:
#                 x = self.bn(x)
#             else:
#                 x = (x - torch.tensor(self.pre_specified_mean_std['mean'],
#                                      device=x.device).unsqueeze(0)) / \
#                     torch.tensor(self.pre_specified_mean_std['std'],
#                                  device=x.device).unsqueeze(0)
#
#             # x = (x - x.mean(0).unsqueeze(0)) / x.std(0).unsqueeze(0)
#         elif self.act_norm == 'all_norm':
#             x = (x - x.mean()) / x.std()
#         elif self.act_norm == 'layer_norm_no_last':
#             if not is_last_node:
#                 x = (x - x.mean(-1).unsqueeze(-1)) / x.std(-1).unsqueeze(-1)
#         elif self.act_norm == 'batch_norm_no_last':
#
#             if not is_last_node:
#                 if self.pre_specified_mean_std is None:
#                     x = self.bn(x)
#                 else:
#                     x = (x - torch.tensor(self.pre_specified_mean_std['mean'],
#                                           device=x.device).unsqueeze(0)) / \
#                         torch.tensor(self.pre_specified_mean_std['std'],
#                                      device=x.device).unsqueeze(0)
#
#         self.x_before_add_noise_second = x.clone()
#
#         if isinstance(self.noise_model_tq, tq.NoiseModelTQActivation):
#             x = self.noise_model_tq.add_noise(x, self.node_id,
#                                               is_after_norm=True)
#
#         self.x_before_act_quant = x.clone()
#
#         return x
#
#     def run_circuit(self, inputs):
#         self.encoder(self.q_device, inputs)
#         self.q_layer(self.q_device)
#         x = self.measure(self.q_device)
#         return x
#
#     def shift_and_run(self, x, use_qiskit=False, is_last_node=False, is_first_node=False, parallel=True):
#         import numpy as np
#         self.circuit_in = x
#         self.circuit_out = None
#         time_spent = datetime.timedelta()
#         if use_qiskit:
#             with torch.no_grad():
#                 bsz = x.shape[0]
#                 inputs = x
#                 x, time_spent_list = self.qiskit_processor.process_parameterized_and_shift(
#                     self.q_device,
#                     self.encoder,
#                     self.q_layer,
#                     self.measure,
#                     inputs,
#                     shift_encoder=False,
#                     parallel=parallel,
#                     shift_this_step=self.shift_this_step)
#                 for ts in time_spent_list:
#                     time_spent = time_spent + ts
#                 results = x.reshape(1 + 2 * np.sum(self.shift_this_step), bsz, self.arch['n_wires'])
#                 self.circuit_out = results[0, :, :].clone()
#
#                 cnt = 0
#                 self.grad_qlayer = []
#                 for i, named_param in enumerate(self.q_layer.named_parameters()):
#                     if self.shift_this_step[i]:
#                         cnt = cnt + 1
#                         out1 = results[cnt,:,:]
#                         cnt = cnt + 1
#                         out2 = results[cnt,:,:]
#                         self.grad_qlayer.append(0.5 * (out1 - out2))
#                     else:
#                         self.grad_qlayer.append(None)
#
#                 self.grad_encoder = []
#                 if not is_first_node:
#                     x, time_spent_list = self.qiskit_processor.process_parameterized_and_shift(
#                         self.q_device,
#                         self.encoder,
#                         self.q_layer,
#                         self.measure,
#                         inputs,
#                         shift_encoder=True,
#                         parallel=parallel)
#                     for ts in time_spent_list:
#                         time_spent = time_spent + ts
#                     results = x.reshape(2 * inputs.shape[1], bsz, self.arch['n_wires'])
#                     cnt = 0
#                     while cnt < 2 * inputs.shape[1]:
#                         out1 = results[cnt,:,:]
#                         cnt = cnt + 1
#                         out2 = results[cnt,:,:]
#                         cnt = cnt + 1
#                         self.grad_encoder.append(0.5 * (out1 - out2))
#         else:
#             with torch.no_grad():
#                 # time1 = datetime.datetime.now()
#                 inputs = x
#                 x = self.run_circuit(inputs)
#                 self.circuit_out = x
#
#                 self.grad_qlayer = []
#                 for i, param in enumerate(self.q_layer.parameters()):
#                     if self.shift_this_step[i]:
#                         param.copy_(param + np.pi * 0.5)
#                         out1 = self.run_circuit(inputs)
#                         param.copy_(param - np.pi)
#                         out2 = self.run_circuit(inputs)
#                         param.copy_(param + np.pi * 0.5)
#                         self.grad_qlayer.append(0.5 * (out1 - out2))
#                     else:
#                         self.grad_qlayer.append(None)
#
#                 self.grad_encoder = []
#                 if not is_first_node:
#                     for input_id in range(inputs.size()[1]):
#                         inputs[:, input_id] += np.pi * 0.5
#                         out1 = self.run_circuit(inputs)
#                         inputs[:, input_id] -= np.pi
#                         out2 = self.run_circuit(inputs)
#                         inputs[:, input_id] += np.pi * 0.5
#                         self.grad_encoder.append(0.5 * (out1 - out2))
#
#                 # time2 = datetime.datetime.now()
#                 # print('one step:')
#                 # print(time2 -time1)
#                 # print('run one circuit:')
#                 # print((time2 -time1) / (1 + 2 * np.sum(self.shift_this_step)))
#
#         x = self.circuit_out
#         self.x_before_add_noise = x.clone()
#         self.circuit_out.requires_grad = True
#
#         if isinstance(self.noise_model_tq, tq.NoiseModelTQActivation):
#             x = self.noise_model_tq.add_noise(x, self.node_id,
#                                               is_after_norm=False)
#
#         self.x_before_norm = x.clone()
#
#         if self.act_norm == 'layer_norm':
#             x = (x - x.mean(-1).unsqueeze(-1)) / x.std(-1).unsqueeze(-1)
#         elif self.act_norm == 'batch_norm':
#             if self.pre_specified_mean_std is None:
#                 x = self.bn(x)
#             else:
#                 x = (x - torch.tensor(self.pre_specified_mean_std['mean'],
#                                      device=x.device).unsqueeze(0)) / \
#                     torch.tensor(self.pre_specified_mean_std['std'],
#                                  device=x.device).unsqueeze(0)
#
#             # x = (x - x.mean(0).unsqueeze(0)) / x.std(0).unsqueeze(0)
#         elif self.act_norm == 'all_norm':
#             x = (x - x.mean()) / x.std()
#         elif self.act_norm == 'layer_norm_no_last':
#             if not is_last_node:
#                 x = (x - x.mean(-1).unsqueeze(-1)) / x.std(-1).unsqueeze(-1)
#         elif self.act_norm == 'batch_norm_no_last':
#
#             if not is_last_node:
#                 if self.pre_specified_mean_std is None:
#                     x = self.bn(x)
#                 else:
#                     x = (x - torch.tensor(self.pre_specified_mean_std['mean'],
#                                           device=x.device).unsqueeze(0)) / \
#                         torch.tensor(self.pre_specified_mean_std['std'],
#                                      device=x.device).unsqueeze(0)
#
#         self.x_before_add_noise_second = x.clone()
#
#         if isinstance(self.noise_model_tq, tq.NoiseModelTQActivation):
#             x = self.noise_model_tq.add_noise(x, self.node_id,
#                                               is_after_norm=True)
#
#         self.x_before_act_quant = x.clone()
#
#         return x, time_spent
#
#
# def build_nodes(node_archs, act_norm=None):
#     nodes = tq.QuantumModuleList()
#     for k, node_arch in enumerate(node_archs):
#         nodes.append(QuantumNode(node_arch, act_norm=act_norm,
#                                  node_id=k))
#
#     return nodes

import argparse
import pdb
import torch
import torchquantum as tq
import numpy as np

from torchpack.utils.logging import logger
from torchquantum.operators import op_name_dict
from torchquantum.functional import func_name_dict
from torchquantum.macro import F_DTYPE
from torchquantum.plugins.qiskit_macros import (
    QISKIT_INCOMPATIBLE_FUNC_NAMES,
    QISKIT_INCOMPATIBLE_OPS,
)


class QLayer(tq.QuantumModule):
    def __init__(
        self,
        n_wires,
        wires,
        n_ops_rd,
        n_ops_cin,
        n_funcs,
        qiskit_compatible=False,
    ):
        super().__init__()
        self.n_wires = n_wires
        self.wires = wires
        self.n_ops_rd = n_ops_rd
        self.n_ops_cin = n_ops_cin
        self.n_funcs = n_funcs
        self.rd_layer = None
        if self.n_ops_rd > 0:
            self.rd_layer = tq.RandomLayerAllTypes(
                n_ops=n_ops_rd, wires=wires, qiskit_compatible=qiskit_compatible
            )

        self.cin_op_types = [
            tq.RX,
            tq.RY,
            tq.RZ,
            tq.PhaseShift,
            tq.Rot,
            tq.MultiRZ,
            tq.CRX,
            tq.CRY,
            tq.CRZ,
            tq.CRot,
            tq.U1,
            tq.U2,
            tq.U3,
        ]
        self.funcs = [
            "hadamard",
            "paulix",
            "pauliy",
            "pauliz",
            "s",
            "t",
            "sx",
            "cnot",
            "cz",
            "cy",
            "rx",
            "ry",
            "rz",
            "swap",
            "cswap",
            "toffoli",
            "phaseshift",
            "rot",
            "multirz",
            "crx",
            "cry",
            "crz",
            "crot",
            "u1",
            "u2",
            "u3",
            "qubitunitary",
            "qubitunitaryfast",
            "multicnot",
            "multixcnot",
        ]

        if qiskit_compatible:
            for op in QISKIT_INCOMPATIBLE_OPS:
                self.cin_op_types.remove(op)
            for func in QISKIT_INCOMPATIBLE_FUNC_NAMES:
                self.funcs.remove(func)

        self.x_idx = 0
        self.cin_op_list = tq.QuantumModuleList()
        self.rand_mat = np.random.randn(2**self.n_wires, 2**self.n_wires)
        self.build_random_cin_layer()

        self.func_list = []
        self.func_wires_list = []
        self.build_random_funcs()

        self.cin_op_inverse = np.random.randint(2, size=n_ops_cin)
        self.func_inverse = np.random.randint(2, size=n_funcs)

    def build_random_cin_layer(self):
        cnt = 0
        while cnt < self.n_ops_cin:
            op = np.random.choice(self.cin_op_types)
            n_op_wires = op.num_wires
            if n_op_wires > self.n_wires:
                continue
            if n_op_wires == -1:
                is_AnyWire = True
                n_op_wires = self.n_wires
            else:
                is_AnyWire = False

            op_wires = list(
                np.random.choice(self.wires, size=n_op_wires, replace=False)
            )

            if is_AnyWire:
                operation = op(n_wires=n_op_wires, wires=op_wires)
            else:
                operation = op(wires=op_wires)

            self.cin_op_list.append(operation)
            cnt += 1

    def build_random_funcs(self):
        cnt = 0
        while cnt < self.n_funcs:
            func = np.random.choice(self.funcs)
            n_func_wires = op_name_dict[func]().num_wires
            if n_func_wires > self.n_wires:
                continue
            cnt += 1

            if n_func_wires == -1:
                n_func_wires = self.n_wires

            func_wires = list(
                np.random.choice(self.wires, size=n_func_wires, replace=False)
            )
            self.func_list.append(func)
            self.func_wires_list.append(func_wires)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device

        # test random generated parameterized layers
        if self.n_ops_rd > 0:
            self.rd_layer(q_device)

        # test layer with classical inputs
        for op, is_inverse in zip(self.cin_op_list, self.cin_op_inverse):
            n_op_params = op.num_params
            if n_op_params == -1:
                n_op_params = 2
            params = x[:, self.x_idx : self.x_idx + n_op_params]
            self.x_idx += n_op_params
            op(self.q_device, params=params, inverse=is_inverse)

        # test functional layers
        for func, func_wires, is_inverse in zip(
            self.func_list, self.func_wires_list, self.func_inverse
        ):
            n_func_wires = len(func_wires)
            n_func_params = op_name_dict[func]().num_params

            if n_func_params == 0:
                if func in ["multicnot", "multixcnot"]:
                    func_name_dict[func](
                        self.q_device,
                        wires=func_wires,
                        n_wires=n_func_wires,
                        static=self.static_mode,
                        parent_graph=self.graph,
                        inverse=is_inverse,
                    )
                else:
                    func_name_dict[func](
                        self.q_device,
                        wires=func_wires,
                        static=self.static_mode,
                        parent_graph=self.graph,
                        inverse=is_inverse,
                    )
            elif n_func_params == -1:
                # qubitunitary:
                if func in ["qubitunitary", "qubitunitaryfast", "qubitunitarystrict"]:
                    u, s, v = np.linalg.svd(
                        self.rand_mat[: 2**n_func_wires, : 2**n_func_wires]
                    )
                    params = u @ v
                    func_name_dict[func](
                        self.q_device,
                        wires=func_wires,
                        n_wires=n_func_wires,
                        params=params,
                        static=self.static_mode,
                        parent_graph=self.graph,
                        inverse=is_inverse,
                    )
                else:
                    raise NotImplementedError
            else:
                params = x[:, self.x_idx : self.x_idx + n_func_params]
                self.x_idx += n_func_params
                if func in ["multirz"]:
                    func_name_dict[func](
                        self.q_device,
                        wires=func_wires,
                        n_wires=n_func_wires,
                        params=params,
                        static=self.static_mode,
                        parent_graph=self.graph,
                        inverse=is_inverse,
                    )
                else:
                    func_name_dict[func](
                        self.q_device,
                        wires=func_wires,
                        params=params,
                        static=self.static_mode,
                        parent_graph=self.graph,
                        inverse=is_inverse,
                    )

        self.x_idx = 0


def static_mode_dynamic_vs_static_gradients_test():
    bsz = 7

    for n_wires in range(2, 10):
        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_dev0 = tq.QuantumDevice(n_wires=n_wires)
        q_dev1 = tq.QuantumDevice(n_wires=n_wires)
        q_dev0.reset_all_eq_states(bsz)
        q_layer = QLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=500,
            n_ops_cin=500,
            n_funcs=500,
        )
        q_layer(q_dev0, x)
        loss = q_dev0.states.abs().sum()
        loss.backward()
        q_layer_dynamic_grads = {}
        for name, params in q_layer.named_parameters():
            q_layer_dynamic_grads[name] = params.grad.data.clone()
            params.grad = None

        for wires_per_block in range(1, n_wires + 1):
            q_dev1.reset_all_eq_states(bsz)
            q_layer.static_off()
            q_layer.static_on(wires_per_block=wires_per_block)
            q_layer(q_dev1, x)
            loss = q_dev1.states.abs().sum()
            loss.backward()
            q_layer_static_grads = {}
            for name, params in q_layer.named_parameters():
                q_layer_static_grads[name] = params.grad.data.clone()
                params.grad = None

            name = None
            try:
                for name, _ in q_layer.named_parameters():
                    diff = q_layer_dynamic_grads[name] - q_layer_static_grads[name]
                    assert torch.allclose(
                        q_layer_dynamic_grads[name],
                        q_layer_static_grads[name],
                        atol=1e-4,
                    )
                    logger.info(f"Diff: {diff}")
                    logger.info(
                        f"PASS: [n_wires]={n_wires}, dynamic VS "
                        f"static run with [wires_per_block]="
                        f"{wires_per_block}. Params: {name}"
                    )
            except AssertionError:
                logger.exception(
                    f"FAIL: [n_wires]={n_wires}, dynamic VS "
                    f"static run with "
                    f"[wires_per_block]={wires_per_block}. "
                    f"Params: {name}"
                )
                raise AssertionError

            q_layer.static_off()


def static_mode_dynamic_vs_static_test():
    bsz = 7

    for n_wires in range(2, 10):
        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_dev0 = tq.QuantumDevice(n_wires=n_wires)
        q_dev1 = tq.QuantumDevice(n_wires=n_wires)
        q_dev0.reset_all_eq_states(bsz)
        q_layer = QLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=500,
            n_ops_cin=500,
            n_funcs=500,
        )
        q_layer(q_dev0, x)
        state_dyna = q_dev0.states.clone()

        for wires_per_block in range(1, n_wires + 1):
            q_dev1.reset_all_eq_states(bsz)
            q_layer.static_off()
            q_layer.static_on(wires_per_block=wires_per_block)
            q_layer(q_dev1, x)
            state_static = q_dev1.states.clone()

            try:
                assert torch.allclose(state_dyna, state_static, atol=1e-5)
                logger.info(
                    f"PASS: [n_wires]={n_wires}, dynamic VS static "
                    f"run with [wires_per_block]={wires_per_block}"
                )
            except AssertionError:
                logger.exception(
                    f"FAIL: [n_wires]={n_wires}, dynamic VS "
                    f"static run with "
                    f"[wires_per_block]={wires_per_block}"
                )
                raise AssertionError

            q_layer.static_off()


def static_mode_dynamic_vs_static_vs_get_unitary_test():
    for n_wires in range(2, 11):
        q_dev0 = tq.QuantumDevice(n_wires=n_wires)
        q_dev1 = tq.QuantumDevice(n_wires=n_wires)
        q_dev2 = tq.QuantumDevice(n_wires=n_wires)
        q_dev0.reset_identity_states()
        q_dev1.reset_identity_states()

        q_layer = tq.RandomLayerAllTypes(n_ops=500, wires=list(range(n_wires)))
        q_layer(q_dev0)
        u_dyna = q_dev0.states.clone().reshape(2**n_wires, 2**n_wires).permute(1, 0)

        u_static_get = q_layer.get_unitary(q_dev1)
        try:
            assert torch.allclose(u_dyna, u_static_get, atol=1e-5)
            logger.info(f"PASS: [n_wires]={n_wires}, dynamic VS get_unitary()")
        except AssertionError:
            logger.exception(f"FAIL: [n_wires]={n_wires}, dynamic VS " f"get_unitary()")
            raise AssertionError

        for wires_per_block in range(1, n_wires + 1):
            q_dev2.reset_identity_states()
            q_layer.static_on(wires_per_block=wires_per_block)
            q_layer(q_dev2)
            u_static_run = (
                q_dev2.states.clone().reshape(2**n_wires, 2**n_wires).permute(1, 0)
            )
            q_layer.static_off()
            try:
                assert torch.allclose(u_dyna, u_static_run, atol=1e-5)
                logger.info(
                    f"PASS: [n_wires]={n_wires}, dynamic VS static "
                    f"run with [wires_per_block]={wires_per_block}"
                )
            except AssertionError:
                logger.exception(
                    f"FAIL: [n_wires]={n_wires}, dynamic VS "
                    f"static run with [wires_per_block]"
                    f"={wires_per_block}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="pdb")
    args = parser.parse_args()

    if args.pdb:
        pdb.set_trace()

    torch.manual_seed(42)
    np.random.seed(42)

    static_mode_dynamic_vs_static_vs_get_unitary_test()
    static_mode_dynamic_vs_static_test()
    static_mode_dynamic_vs_static_gradients_test()
