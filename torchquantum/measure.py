import torch
import torchquantum as tq
import numpy as np

from typing import Union, List


def expval(q_device: tq.QuantumDevice,
           wires: Union[int, List[int]],
           observables: Union[tq.Observable, List[tq.Observable]]):

    all_dims = np.arange(q_device.states.dim())
    if isinstance(wires, int):
        wires = [wires]
        observables = [observables]

    # rotation to the desired basis
    for wire, observable in zip(wires, observables):
        for rotation in observable.diagonalizing_gates():
            rotation(q_device, wires=wire)

    states = q_device.states
    # compute magnitude
    state_mag = torch.abs(states) ** 2

    expectations = []
    for wire, observable in zip(wires, observables):
        # compute marginal magnitude
        reduction_dims = np.delete(all_dims, [0, wire + 1])
        probs = state_mag.sum(list(reduction_dims))
        res = probs.mv(observable.eigvals.real.to(probs.device))
        expectations.append(res)

    return torch.stack(expectations, dim=-1)


class MeasureAll(tq.QuantumModule):
    def __init__(self, obs, v_c_reg_mapping=None):
        super().__init__()
        self.obs = obs
        self.v_c_reg_mapping = v_c_reg_mapping

    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        x = expval(q_device, list(range(q_device.n_wires)), [self.obs()] *
                   q_device.n_wires)

        if self.v_c_reg_mapping is not None:
            c2v_mapping = self.v_c_reg_mapping['c2v']
            """
            the measurement is not normal order, need permutation 
            """
            perm = []
            for k in range(x.shape[-1]):
                if k in c2v_mapping.keys():
                    perm.append(c2v_mapping[k])
            x = x[:, perm]

        if self.noise_model_tq is not None and \
                self.noise_model_tq.is_add_noise:
            # add readout error
            noise_free_0_probs = (x + 1) / 2
            noise_free_1_probs = 1 - (x + 1) / 2

            p_c_reg_mapping = self.noise_model_tq.p_c_reg_mapping
            c2p_mapping = p_c_reg_mapping['c2p']

            measure_info = self.noise_model_tq.parsed_dict['measure']
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
        else:
            return x

    def set_v_c_reg_mapping(self, mapping):
        self.v_c_reg_mapping = mapping


class MeasureMultipleTimes(tq.QuantumModule):
    """
    obs list:
    list of dict: example
    {'wires': [0, 2, 3, 1], 'observables': ['x', 'y', 'z', 'i']
    }
    """
    def __init__(self, obs_list, v_c_reg_mapping=None):
        super().__init__()
        self.obs_list = obs_list
        self.v_c_reg_mapping = v_c_reg_mapping

    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        res_all = []

        for layer in self.obs_list:
            # create a new q device for each time of measurement
            q_device_new = tq.QuantumDevice(n_wires=q_device.n_wires)
            q_device_new.clone_states(existing_states=q_device.states)
            q_device_new.state = q_device.state

            observables = []
            for wire in range(q_device.n_wires):
                observables.append(tq.I())

            for wire, observable in zip(layer['wires'], layer['observables']):
                observables[wire] = tq.op_name_dict[observable]()

            res = expval(q_device_new, wires=list(range(q_device.n_wires)),
                         observables=observables)

            if self.v_c_reg_mapping is not None:
                c2v_mapping = self.v_c_reg_mapping['c2v']
                """
                the measurement is not normal order, need permutation 
                """
                perm = []
                for k in range(res.shape[-1]):
                    if k in c2v_mapping.keys():
                        perm.append(c2v_mapping[k])
                res = res[:, perm]
            res_all.append(res)

        return torch.cat(res_all)

    def set_v_c_reg_mapping(self, mapping):
        self.v_c_reg_mapping = mapping
