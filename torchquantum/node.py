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

        return x


def build_nodes(node_archs):
    nodes = tq.QuantumModuleList()
    for node_arch in node_archs:
        nodes.append(QuantumNode(node_arch))

    return nodes
