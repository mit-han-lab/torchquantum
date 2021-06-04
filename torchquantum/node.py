import torch
import torchquantum as tq

from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.layers import layer_name_dict


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

    def forward(self, x, use_qiskit=False, is_last_node=False):
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device,
                self.encoder,
                self.q_layer,
                self.measure,
                x)
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


def build_nodes(node_archs, act_norm=None):
    nodes = tq.QuantumModuleList()
    for k, node_arch in enumerate(node_archs):
        nodes.append(QuantumNode(node_arch, act_norm=act_norm,
                                 node_id=k))

    return nodes
