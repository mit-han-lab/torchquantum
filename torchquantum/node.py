import torch
import torchquantum as tq

from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.layers import layer_name_dict
import numpy as np
import datetime


__all__ = ['QuantumNode',
           'build_nodes',
           ]


class QuantumNode(tq.QuantumModule):
    """
    a quantum node contains a q device, encoder, q layer and measure
    """
    def __init__(self, arch, act_norm, node_id):
        super().__init__()
        self.arch = arch
        self.q_device = tq.QuantumDevice(n_wires=arch['n_wires'])
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict[
                                             arch['encoder_op_list_name']])
        self.q_layer = layer_name_dict[arch['q_layer_name']](arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.act_norm = act_norm
        self.x_before_add_noise = None
        self.x_before_add_noise_second = None
        self.x_before_act_quant = None
        self.x_before_norm = None
        self.circuit_in = None
        self.circuit_out = None
        self.shift_this_step = np.array([True] * len(list(self.q_layer.parameters())))
        self.cool_down = [0] * len(list(self.q_layer.parameters()))
        self.triger_cd = [0] * len(list(self.q_layer.parameters()))
        if self.act_norm == 'batch_norm' or \
                self.act_norm == 'batch_norm_no_last':
            self.bn = torch.nn.BatchNorm1d(
                num_features=arch['n_wires'],
                momentum=None,
                affine=False,
                track_running_stats=False
            )
        self.node_id = node_id
        self.pre_specified_mean_std = None
        self.grad_qlayer = None
        self.grad_encoder = None

    def forward(self, x, use_qiskit=False, is_last_node=False, parallel=True):
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device,
                self.encoder,
                self.q_layer,
                self.measure,
                x,
                parallel=parallel
            )
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        self.x_before_add_noise = x.clone()

        if isinstance(self.noise_model_tq, tq.NoiseModelTQActivation):
            x = self.noise_model_tq.add_noise(x, self.node_id,
                                              is_after_norm=False)

        self.x_before_norm = x.clone()

        if self.act_norm == 'layer_norm':
            x = (x - x.mean(-1).unsqueeze(-1)) / x.std(-1).unsqueeze(-1)
        elif self.act_norm == 'batch_norm':
            if self.pre_specified_mean_std is None:
                x = self.bn(x)
            else:
                x = (x - torch.tensor(self.pre_specified_mean_std['mean'],
                                     device=x.device).unsqueeze(0)) / \
                    torch.tensor(self.pre_specified_mean_std['std'],
                                 device=x.device).unsqueeze(0)

            # x = (x - x.mean(0).unsqueeze(0)) / x.std(0).unsqueeze(0)
        elif self.act_norm == 'all_norm':
            x = (x - x.mean()) / x.std()
        elif self.act_norm == 'layer_norm_no_last':
            if not is_last_node:
                x = (x - x.mean(-1).unsqueeze(-1)) / x.std(-1).unsqueeze(-1)
        elif self.act_norm == 'batch_norm_no_last':

            if not is_last_node:
                if self.pre_specified_mean_std is None:
                    x = self.bn(x)
                else:
                    x = (x - torch.tensor(self.pre_specified_mean_std['mean'],
                                          device=x.device).unsqueeze(0)) / \
                        torch.tensor(self.pre_specified_mean_std['std'],
                                     device=x.device).unsqueeze(0)

        self.x_before_add_noise_second = x.clone()

        if isinstance(self.noise_model_tq, tq.NoiseModelTQActivation):
            x = self.noise_model_tq.add_noise(x, self.node_id,
                                              is_after_norm=True)

        self.x_before_act_quant = x.clone()

        return x

    def run_circuit(self, inputs):
        self.encoder(self.q_device, inputs)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)
        return x

    def shift_and_run(self, x, use_qiskit=False, is_last_node=False, is_first_node=False, parallel=True):
        import numpy as np
        self.circuit_in = x
        self.circuit_out = None
        time_spent = datetime.timedelta()
        if use_qiskit:
            with torch.no_grad():
                bsz = x.shape[0]
                inputs = x
                x, time_spent_list = self.qiskit_processor.process_parameterized_and_shift(
                    self.q_device,
                    self.encoder,
                    self.q_layer,
                    self.measure,
                    inputs,
                    shift_encoder=False,
                    parallel=parallel,
                    shift_this_step=self.shift_this_step)
                for ts in time_spent_list:
                    time_spent = time_spent + ts
                results = x.reshape(1 + 2 * np.sum(self.shift_this_step), bsz, self.arch['n_wires'])
                self.circuit_out = results[0, :, :].clone()

                cnt = 0
                self.grad_qlayer = []
                for i, named_param in enumerate(self.q_layer.named_parameters()):
                    if self.shift_this_step[i]:
                        cnt = cnt + 1
                        out1 = results[cnt,:,:]
                        cnt = cnt + 1
                        out2 = results[cnt,:,:]
                        self.grad_qlayer.append(0.5 * (out1 - out2))
                    else:
                        self.grad_qlayer.append(None)
                
                self.grad_encoder = []
                if not is_first_node:
                    x, time_spent_list = self.qiskit_processor.process_parameterized_and_shift(
                        self.q_device,
                        self.encoder,
                        self.q_layer,
                        self.measure,
                        inputs,
                        shift_encoder=True,
                        parallel=parallel)
                    for ts in time_spent_list:
                        time_spent = time_spent + ts
                    results = x.reshape(2 * inputs.shape[1], bsz, self.arch['n_wires'])
                    cnt = 0
                    while cnt < 2 * inputs.shape[1]:
                        out1 = results[cnt,:,:]
                        cnt = cnt + 1
                        out2 = results[cnt,:,:]
                        cnt = cnt + 1
                        self.grad_encoder.append(0.5 * (out1 - out2))
        else:
            with torch.no_grad():
                # time1 = datetime.datetime.now()
                inputs = x
                x = self.run_circuit(inputs)
                self.circuit_out = x

                self.grad_qlayer = []
                for i, param in enumerate(self.q_layer.parameters()):
                    if self.shift_this_step[i]:
                        param.copy_(param + np.pi * 0.5)
                        out1 = self.run_circuit(inputs)
                        param.copy_(param - np.pi)
                        out2 = self.run_circuit(inputs)
                        param.copy_(param + np.pi * 0.5)
                        self.grad_qlayer.append(0.5 * (out1 - out2))
                    else:
                        self.grad_qlayer.append(None)
                
                self.grad_encoder = []
                if not is_first_node:
                    for input_id in range(inputs.size()[1]):
                        inputs[:, input_id] += np.pi * 0.5
                        out1 = self.run_circuit(inputs)
                        inputs[:, input_id] -= np.pi
                        out2 = self.run_circuit(inputs)
                        inputs[:, input_id] += np.pi * 0.5
                        self.grad_encoder.append(0.5 * (out1 - out2))
                
                # time2 = datetime.datetime.now()
                # print('one step:')
                # print(time2 -time1)
                # print('run one circuit:')
                # print((time2 -time1) / (1 + 2 * np.sum(self.shift_this_step)))

        x = self.circuit_out
        self.x_before_add_noise = x.clone()
        self.circuit_out.requires_grad = True

        if isinstance(self.noise_model_tq, tq.NoiseModelTQActivation):
            x = self.noise_model_tq.add_noise(x, self.node_id,
                                              is_after_norm=False)

        self.x_before_norm = x.clone()

        if self.act_norm == 'layer_norm':
            x = (x - x.mean(-1).unsqueeze(-1)) / x.std(-1).unsqueeze(-1)
        elif self.act_norm == 'batch_norm':
            if self.pre_specified_mean_std is None:
                x = self.bn(x)
            else:
                x = (x - torch.tensor(self.pre_specified_mean_std['mean'],
                                     device=x.device).unsqueeze(0)) / \
                    torch.tensor(self.pre_specified_mean_std['std'],
                                 device=x.device).unsqueeze(0)

            # x = (x - x.mean(0).unsqueeze(0)) / x.std(0).unsqueeze(0)
        elif self.act_norm == 'all_norm':
            x = (x - x.mean()) / x.std()
        elif self.act_norm == 'layer_norm_no_last':
            if not is_last_node:
                x = (x - x.mean(-1).unsqueeze(-1)) / x.std(-1).unsqueeze(-1)
        elif self.act_norm == 'batch_norm_no_last':

            if not is_last_node:
                if self.pre_specified_mean_std is None:
                    x = self.bn(x)
                else:
                    x = (x - torch.tensor(self.pre_specified_mean_std['mean'],
                                          device=x.device).unsqueeze(0)) / \
                        torch.tensor(self.pre_specified_mean_std['std'],
                                     device=x.device).unsqueeze(0)

        self.x_before_add_noise_second = x.clone()

        if isinstance(self.noise_model_tq, tq.NoiseModelTQActivation):
            x = self.noise_model_tq.add_noise(x, self.node_id,
                                              is_after_norm=True)

        self.x_before_act_quant = x.clone()

        return x, time_spent


def build_nodes(node_archs, act_norm=None):
    nodes = tq.QuantumModuleList()
    for k, node_arch in enumerate(node_archs):
        nodes.append(QuantumNode(node_arch, act_norm=act_norm,
                                 node_id=k))

    return nodes
