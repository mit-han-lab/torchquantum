import torchquantum as tq
import torchquantum.functional as tqf
import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import datetime

from torchquantum.encoding import encoder_op_list_name_dict
from torchpack.utils.logging import logger
from torchquantum.layers import layer_name_dict
from torchpack.utils.config import configs


class QMultiFCModel0(tq.QuantumModule):
    # multiple nodes, one node contains encoder, q_layer, and measure
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.n_nodes = arch['n_nodes']
        self.nodes = tq.build_nodes(arch['node_archs'], act_norm=arch['act_norm'])
        assert arch['n_nodes'] == len(arch['node_archs'])
        self.mse_all = []
        self.residual = getattr(arch, 'residual', False)
        self.activations = []
        self.work_from_step = 0
        self.grad_dict = None
        self.count1 = 0
        self.count2 = 0
        self.num_forwards = 0
        self.n_params = len(list(self.nodes[0].parameters()))
        self.last_abs_grad = torch.zeros(self.n_params)
        self.pruning_method = arch['pruning_method']
        if self.pruning_method == 'random_pruning':
            self.sampling_ratio = 1 - arch['pruning_ratio']
        elif self.pruning_method == 'perlayer_pruning':
            self.n_qubits = arch['node_archs'][0]['n_wires']
            self.n_layers = arch['node_archs'][0]['n_layers_per_block'] * arch['node_archs'][0]['n_blocks']
            self.n_sampling_layers = self.n_layers - arch['n_pruning_layers']
            self.colums = np.arange(self.n_sampling_layers)
        elif self.pruning_method == 'perqubit_pruning':
            self.n_qubits = arch['node_archs'][0]['n_wires']
            self.n_layers = arch['node_archs'][0]['n_layers_per_block'] * arch['node_archs'][0]['n_blocks']
            self.n_sampling_qubits = self.n_qubits - arch['n_pruning_qubits']
            self.rows = np.arange(self.n_sampling_qubits)
        elif self.pruning_method == 'gradient_based_pruning':
            self.accumulation_window_size = arch['accumulation_window_size']
            self.pruning_window_size = arch['pruning_window_size']
            self.sampling_ratio = 1 - arch['pruning_ratio']
            self.sum_abs_grad = torch.tensor([0.01] * self.n_params)
            self.is_accumulation = True
            self.accumulation_steps = 0
            self.pruning_steps = 0
        elif self.pruning_method == 'gradient_based_deterministic':
            self.accumulation_window_size = arch['accumulation_window_size']
            self.pruning_window_size = arch['pruning_window_size']
            self.sampling_ratio = 1 - arch['pruning_ratio']
            self.sum_abs_grad = torch.tensor([0.01] * self.n_params)
            self.is_accumulation = True
            self.accumulation_steps = 0
            self.pruning_steps = 0
        elif self.pruning_method == 'phase_based_pruning':
            self.accumulation_window_size = arch['accumulation_window_size']
            self.pruning_window_size = arch['pruning_window_size']
            self.sampling_ratio = 1 - arch['pruning_ratio']
            self.last_abs_param = torch.zeros(self.n_params)
            self.is_accumulation = True
            self.accumulation_steps = 0
            self.pruning_steps = 0
        else:
            logger.info('Not use any pruning')

    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]

        if getattr(self.arch, 'down_sample_kernel_size', None) is not None:
            x = F.avg_pool2d(x, self.arch['down_sample_kernel_size'])

        if getattr(self.arch, 'fft_remain_size', None) is not None:
            x = torch.fft.fft2(x, norm='ortho').abs()[:, :,
                :self.arch['fft_remain_size'], :self.arch[
                'fft_remain_size']]
            x = x.contiguous()

        x = x.view(bsz, -1)
        mse_all = []

        for k, node in enumerate(self.nodes):
            node_out = node(x,
                            use_qiskit=use_qiskit,
                            is_last_node=(k == self.n_nodes - 1))
            x = node_out
        self.mse_all = mse_all

        if getattr(self.arch, 'output_len', None) is not None:
            x = x.reshape(bsz, -1, self.arch.output_len).sum(-1)

        if x.dim() > 2:
            x = x.squeeze()

        x = F.log_softmax(x, dim=1)
        return x

    def shift_and_run(self, x, global_step, total_step, verbose=False, use_qiskit=False):
        bsz = x.shape[0]

        if getattr(self.arch, 'down_sample_kernel_size', None) is not None:
            x = F.avg_pool2d(x, self.arch['down_sample_kernel_size'])

        if getattr(self.arch, 'fft_remain_size', None) is not None:
            x = torch.fft.fft2(x, norm='ortho').abs()[:, :,
                :self.arch['fft_remain_size'], :self.arch[
                'fft_remain_size']]
            x = x.contiguous()

        x = x.view(bsz, -1)
        mse_all = []
        
        for k, node in enumerate(self.nodes):
            node.shift_this_step[:] = True
            if self.pruning_method == 'random_pruning':
                node.shift_this_step[:] = False
                idx = torch.randperm(self.n_params)[:int(self.sampling_ratio * self.n_params)]
                node.shift_this_step[idx] = True
            elif self.pruning_method == 'perlayer_pruning':
                node.shift_this_step[:] = False
                idxs = torch.range(0, self.n_params-1, dtype=int).view(self.n_qubits, self.n_layers)
                sampled_colums = self.colums
                for colum in sampled_colums:
                    node.shift_this_step[idxs[:, colum]] = True
                self.colums += self.n_sampling_layers
                self.colums %= self.n_layers
            elif self.pruning_method == 'perqubit_pruning':
                node.shift_this_step[:] = False
                idxs = torch.range(0, self.n_params-1, dtype=int).view(self.n_qubits, self.n_layers)
                sampled_rows = self.rows
                for row in sampled_rows:
                    node.shift_this_step[idxs[row]] = True
                self.rows += self.n_sampling_qubits
                self.rows %= self.n_qubits
            elif self.pruning_method == 'gradient_based_pruning':
                if self.is_accumulation:
                    self.accumulation_steps += 1
                    self.sum_abs_grad = self.sum_abs_grad + self.last_abs_grad
                    node.shift_this_step[:] = True
                    if self.accumulation_steps == self.accumulation_window_size:
                        self.is_accumulation = False
                        self.accumulation_steps = 0
                        self.sum_abs_grad = torch.tensor([0.01] * self.n_params)
                else:
                    self.pruning_steps += 1
                    node.shift_this_step[:] = False
                    idx = torch.multinomial(self.sum_abs_grad, int(self.sampling_ratio * self.n_params))
                    node.shift_this_step[idx] = True
                    if self.pruning_steps == self.pruning_window_size:
                        self.is_accumulation = True
                        self.pruning_steps = 0
            elif self.pruning_method == 'gradient_based_deterministic':
                if self.is_accumulation:
                    self.accumulation_steps += 1
                    self.sum_abs_grad = self.sum_abs_grad + self.last_abs_grad
                    node.shift_this_step[:] = True
                    if self.accumulation_steps == self.accumulation_window_size:
                        self.is_accumulation = False
                        self.accumulation_steps = 0
                else:
                    self.pruning_steps += 1
                    node.shift_this_step[:] = False
                    idx = torch.argsort(self.sum_abs_grad, descending=True)[:int(self.sampling_ratio * self.n_params)]
                    # idx = torch.multinomial(self.sum_abs_grad, int(self.sampling_ratio * self.n_params))
                    node.shift_this_step[idx] = True
                    if self.pruning_steps == self.pruning_window_size:
                        self.is_accumulation = True
                        self.pruning_steps = 0
                        self.sum_abs_grad = torch.tensor([0.01] * self.n_params)
            elif self.pruning_method == 'phase_based_pruning':
                if self.is_accumulation:
                    self.accumulation_steps += 1
                    node.shift_this_step[:] = True
                    if self.accumulation_steps == self.accumulation_window_size:
                        self.is_accumulation = False
                        self.accumulation_steps = 0
                else:
                    self.pruning_steps += 1
                    node.shift_this_step[:] = False
                    for i, param in enumerate(self.parameters()):
                        param_item = param.item()
                        while param_item > np.pi:
                            param_item -= 2 * np.pi
                        while param_item < - np.pi:
                            param_item += 2 * np.pi
                        self.last_abs_param[i] = 0.01 + np.abs(param_item)
                    idx = torch.multinomial(self.last_abs_param, int(self.sampling_ratio * self.n_params))
                    node.shift_this_step[idx] = True
                    if self.pruning_steps == self.pruning_window_size:
                        self.is_accumulation = True
                        self.pruning_steps = 0
            
            self.num_forwards += 1 + 2 * np.sum(node.shift_this_step)
            node_out, time_spent = node.shift_and_run(x,
                            use_qiskit=use_qiskit,
                            is_last_node=(k == self.n_nodes - 1),
                            is_first_node=(k == 0),
                            parallel=False)
            # logger.info('Time spent:')
            # logger.info(time_spent)
            x = node_out
            mse_all.append(F.mse_loss(node_out, node.x_before_act_quant))
            if verbose:
                acts = {
                    'x_before_add_noise': node.x_before_add_noise.cpu(
                        ).detach().data,
                    'x_before_norm': node.x_before_norm.cpu().detach().data,
                    'x_before_add_noise_second':
                        node.x_before_add_noise_second.cpu().detach().data,
                    'x_before_act_quant': node.x_before_act_quant.cpu().detach(
                        ).data,
                    'x_after_act_quant': node_out.cpu().detach().data,
                }
                self.activations.append(acts)

        self.mse_all = mse_all

        if verbose:
            os.makedirs(os.path.join(configs.run_dir, 'activations'),
                        exist_ok=True)
            torch.save(self.activations,
                       os.path.join(configs.run_dir, 'activations',
                                    f"{configs.eval_config_dir}.pt"))
            # logger.info(f"[use_qiskit]={use_qiskit},
            # expectation:\n {x.data}")

        if getattr(self.arch, 'output_len', None) is not None:
            x = x.reshape(bsz, -1, self.arch.output_len).sum(-1)

        if x.dim() > 2:
            x = x.squeeze()

        x = F.log_softmax(x, dim=1)
        return x
    
    def backprop_grad(self):
        for k, node in reversed(list(enumerate(self.nodes))):
            grad_output = node.circuit_out.grad
            for i, param in enumerate(node.q_layer.parameters()):
                if node.shift_this_step[i]:
                    param.grad = torch.sum(node.grad_qlayer[i] * grad_output).to(dtype=torch.float32).view(param.shape)
                else:
                    self.count1 = self.count1 + 1
                    param.grad = torch.tensor(0.).to(dtype=torch.float32, device=param.device).view(param.shape)
                self.last_abs_grad[i] = np.abs(param.grad.item())
                # if (np.abs(param.grad.item()) < 0):
                #     param.grad = torch.tensor(0.).to(dtype=torch.float32, device=param.device).view(param.shape)
                #     self.count1 = self.count1 + 1
                self.count2 = self.count2 + 1
            
            inputs_grad2loss = None
            for input_grad in node.grad_encoder:
                input_grad2loss = torch.sum(input_grad * grad_output, dim=1).view(-1, 1)
                if inputs_grad2loss == None:
                    inputs_grad2loss = input_grad2loss
                else:
                    inputs_grad2loss = torch.cat((inputs_grad2loss, input_grad2loss), 1)
            
            if k != 0:
                node.circuit_in.backward(inputs_grad2loss)
        # logger.info(str(self.count1) + '/' + str(self.count2))


model_dict = {
    'q_multifc0': QMultiFCModel0,
}
