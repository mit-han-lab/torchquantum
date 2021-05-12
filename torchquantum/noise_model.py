import numpy as np
import torchquantum as tq

from torchpack.utils.logging import logger
from qiskit.providers.aer.noise import NoiseModel
from torchquantum.utils import get_provider

__all__ = ['NoiseModelTQ']


class NoiseModelTQ(object):
    def __init__(self, backend_name, noise_total_prob=None):
        self.backend_name = backend_name
        provider = get_provider(backend_name=backend_name)
        backend = provider.get_backend(backend_name)

        self.noise_model = NoiseModel.from_backend(backend)
        self.noise_model_dict = self.noise_model.to_dict()
        self.is_add_noise = False
        self.v_c_reg_mapping = None
        self.p_c_reg_mapping = None
        self.p_v_reg_mapping = None
        self.noise_total_prob = noise_total_prob

        self.parsed_dict = self.parse_noise_model_dict(self.noise_model_dict)
        self.parsed_dict = self.clean_parsed_noise_model_dict(self.parsed_dict)

    @staticmethod
    def clean_parsed_noise_model_dict(nm_dict):
        # remove the kraus and id operation in the instructions and probs
        for operation, operation_info in nm_dict.items():
            for qubit, qubit_info in operation_info.items():
                inst_all = []
                prob_all = []
                if qubit_info['type'] == 'qerror':
                    for inst, prob in zip(qubit_info['instructions'],
                                          qubit_info['probabilities']):
                        if len(inst) == 1 and inst[0]['name'] in ['id',
                                                                  'kraus']:
                            continue
                        else:
                            inst_all.append(inst)
                            prob_all.append(prob)
                    nm_dict[operation][qubit]['instructions'] = inst_all
                    nm_dict[operation][qubit]['probabilities'] = prob_all
        return nm_dict

    @staticmethod
    def parse_noise_model_dict(nm_dict):
        # the qubits here are physical (p) qubits
        parsed = {}
        nm_list = nm_dict['errors']

        for info in nm_list:
            val_dict = {
                'type': info['type'],
                'instructions': info.get('instructions', None),
                'probabilities': info['probabilities'],
            }

            if info['operations'][0] not in parsed.keys():
                parsed[info['operations'][0]] = {
                    tuple(info['gate_qubits'][0]): val_dict
                }
            elif tuple(info['gate_qubits'][0]) not in parsed[
                    info['operations'][0]].keys():
                parsed[info['operations'][0]][tuple(info['gate_qubits'][0])]\
                    = val_dict
            else:
                raise ValueError

        return parsed

    def magnify_probs(self, probs):
        if self.noise_total_prob is not None:
            factor = self.noise_total_prob / sum(probs)
            probs = [prob * factor for prob in probs]

        return probs

    def sample_noise_op(self, op_in):
        op_name = op_in.name.lower()
        if op_name == 'paulix':
            op_name = 'x'
        elif op_name == 'cnot':
            op_name = 'cx'
        elif op_name in ['sx', 'id']:
            pass
        elif op_name == 'rz':
            # no noise
            return []
        else:
            logger.warning(f"No noise model for {op_name} operation!")

        wires = op_in.wires

        p_wires = [self.p_v_reg_mapping['v2p'][wire] for wire in wires]

        inst_prob = self.parsed_dict[op_name][tuple(p_wires)]

        inst = inst_prob['instructions']
        if len(inst) == 0:
            return []

        probs = inst_prob['probabilities']
        magnified_probs = self.magnify_probs(probs)

        idx = np.random.choice(list(range(len(inst) + 1)),
                               p=magnified_probs + [1 - sum(magnified_probs)])
        if idx == len(inst):
            return []

        instructions = inst[idx]

        ops = []
        for instruction in instructions:
            v_wires = [self.p_v_reg_mapping['p2v'][qubit] for qubit in
                       instruction['qubits']]
            if instruction['name'] == 'x':
                op = tq.PauliX(wires=v_wires)
            elif instruction['name'] == 'y':
                op = tq.PauliY(wires=v_wires)
            elif instruction['name'] == 'z':
                op = tq.PauliZ(wires=v_wires)
            elif instruction['name'] == 'reset':
                op = tq.Reset(wires=v_wires)
            else:
                # ignore id and kraus operation
                # logger.warning(f"skip noise operation {instruction['name']}")
                continue
            ops.append(op)

        return ops
