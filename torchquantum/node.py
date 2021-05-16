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
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.q_device = tq.QuantumDevice(n_wires=arch['n_wires'])
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict[
                                             arch['encoder_op_list_name']])
        self.q_layer = layer_name_dict[arch['q_layer_name']](arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.act_norm = arch.get('act_norm', None)
        self.x_before_act_quant = None
        self.x_before_norm = None

    def forward(self, x, use_qiskit=False):
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

        self.x_before_norm = x.clone()

        if self.act_norm == 'layer_norm':
            x = (x - x.mean(-1).unsqueeze(-1)) / x.std(-1).unsqueeze(-1)
        elif self.act_norm == 'batch_norm':
            x = (x - x.mean(0).unsqueeze(0)) / x.std(0).unsqueeze(0)
        elif self.act_norm == 'all_norm':
            x = (x - x.mean()) / x.std()

        self.x_before_act_quant = x.clone()

        return x


def build_nodes(node_archs):
    nodes = tq.QuantumModuleList()
    for node_arch in node_archs:
        nodes.append(QuantumNode(node_arch))

    return nodes
