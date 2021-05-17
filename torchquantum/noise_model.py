import numpy as np
import torch
import torchquantum as tq

from torchpack.utils.logging import logger
from qiskit.providers.aer.noise import NoiseModel
from torchquantum.utils import get_provider


__all__ = ['NoiseModelTQ',
           'NoiseModelTQActivation',
           'NoiseModelTQPhase',
           ]


class NoiseModelTQ(object):
    def __init__(self,
                 noise_model_name,
                 n_epochs,
                 noise_total_prob=None,
                 ignored_ops=('id', 'kraus', 'reset'),
                 ):
        self.noise_model_name = noise_model_name
        provider = get_provider(backend_name=noise_model_name)
        backend = provider.get_backend(noise_model_name)

        self.noise_model = NoiseModel.from_backend(backend)
        self.noise_model_dict = self.noise_model.to_dict()
        self.is_add_noise = True
        self.v_c_reg_mapping = None
        self.p_c_reg_mapping = None
        self.p_v_reg_mapping = None
        self.orig_noise_total_prob = noise_total_prob
        self.noise_total_prob = noise_total_prob
        self.mode = 'train'
        self.ignored_ops = ignored_ops

        self.parsed_dict = self.parse_noise_model_dict(self.noise_model_dict)
        self.parsed_dict = self.clean_parsed_noise_model_dict(
            self.parsed_dict, ignored_ops)
        self.n_epochs = n_epochs

    def adjust_noise(self, current_epoch):
        self.noise_total_prob = self.orig_noise_total_prob

    @staticmethod
    def clean_parsed_noise_model_dict(nm_dict, ignored_ops):
        # remove the ignored operation in the instructions and probs
        for operation, operation_info in nm_dict.items():
            for qubit, qubit_info in operation_info.items():
                inst_all = []
                prob_all = []
                if qubit_info['type'] == 'qerror':
                    for inst, prob in zip(qubit_info['instructions'],
                                          qubit_info['probabilities']):
                        if any([inst_one['name'] in ignored_ops for inst_one
                                in inst]):
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
        factor = self.noise_total_prob / sum(probs)
        probs = [prob * factor for prob in probs]

        return probs

    def sample_noise_op(self, op_in):
        if not (self.mode == 'train' and self.is_add_noise):
            return []

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

        if self.noise_total_prob is not None:
            magnified_probs = self.magnify_probs(probs)
        else:
            magnified_probs = probs

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
                # ignore operations specified by self.ignored_ops
                # logger.warning(f"skip noise operation {instruction['name']}")
                continue
            ops.append(op)

        return ops

    def apply_readout_error(self, x):
        # add readout error
        noise_free_0_probs = (x + 1) / 2
        noise_free_1_probs = 1 - (x + 1) / 2

        c2p_mapping = self.p_c_reg_mapping['c2p']

        measure_info = self.parsed_dict['measure']
        noisy_0_to_0_prob_all = []
        noisy_0_to_1_prob_all = []
        noisy_1_to_0_prob_all = []
        noisy_1_to_1_prob_all = []

        for k in range(x.shape[-1]):
            p_wire = [c2p_mapping[k]]
            noisy_0_to_0_prob_all.append(measure_info[tuple(p_wire)][
                                             'probabilities'][0][0])
            noisy_0_to_1_prob_all.append(measure_info[tuple(p_wire)][
                                             'probabilities'][0][1])
            noisy_1_to_0_prob_all.append(measure_info[tuple(p_wire)][
                                             'probabilities'][1][0])
            noisy_1_to_1_prob_all.append(measure_info[tuple(p_wire)][
                                             'probabilities'][1][1])

        noisy_0_to_0_prob_all = torch.tensor(noisy_0_to_0_prob_all,
                                             device=x.device)
        noisy_0_to_1_prob_all = torch.tensor(noisy_0_to_1_prob_all,
                                             device=x.device)
        noisy_1_to_0_prob_all = torch.tensor(noisy_1_to_0_prob_all,
                                             device=x.device)
        noisy_1_to_1_prob_all = torch.tensor(noisy_1_to_1_prob_all,
                                             device=x.device)

        noisy_measured_0 = noise_free_0_probs * noisy_0_to_0_prob_all + \
            noise_free_1_probs * noisy_1_to_0_prob_all

        noisy_measured_1 = noise_free_0_probs * noisy_0_to_1_prob_all + \
            noise_free_1_probs * noisy_1_to_1_prob_all
        noisy_expectation = noisy_measured_0 * 1 + noisy_measured_1 * (-1)

        return noisy_expectation


class NoiseModelTQActivation(object):
    """
    add noise to the activations
    """
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
        self.is_add_noise = True
        self.mode = 'train'
        self.noise_total_prob = self.std

    def adjust_noise(self, current_epoch):
        pass

    def sample_noise_op(self, op_in):
        return []

    def apply_readout_error(self, x):
        return x

    def add_noise(self, x):
        if self.mode == 'train' and self.is_add_noise:
            x = x + torch.randn(x.shape, device=x.device) * self.std + \
                self.mean

        return x


class NoiseModelTQPhase(object):
    """
    add noise to rotation parameters
    """
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
        self.is_add_noise = True
        self.mode = 'train'
        self.noise_total_prob = self.std

    def adjust_noise(self, current_epoch):
        pass

    def sample_noise_op(self, op_in):
        return []

    def apply_readout_error(self, x):
        return x

    def add_noise(self, phase):
        if self.mode == 'train' and self.is_add_noise:
            phase = phase +  torch.randn(phase.shape, device=phase.device) * \
                       self.std * np.pi + self.mean

        return phase
