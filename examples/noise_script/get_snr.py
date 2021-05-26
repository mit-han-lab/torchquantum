from torchpack.utils.logging import logger
import argparse
from matplotlib.ticker import FormatStrFormatter

import torch

import matplotlib.pyplot as plt
import os
import numpy as np


import numpy as np
from scipy.ndimage.filters import gaussian_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', type=str)
    parser.add_argument('--noisy', type=str)
    parser.add_argument('--pdb', action='store_true')
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--drawfour', action='store_true')
    parser.add_argument('--path', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    # parser.add_argument('--')

    args = parser.parse_args()
    print(args)

    valid = '_valid' if args.valid else ''

    last_step_dict = {
        'mnist': {
            'four0123': 18400,
            'two36': 9000,
        },
        'fashion': {
            'four0123': 18000,
            'two36': 9000,
        },
        'vowel': {
            'four0516': 10400,
        }
    }

    if args.pdb:
        import pdb
        pdb.set_trace()

    devices = [
        'santiago',
        'belem',
        'lima',
        'quito',
        'athens',
    ]
    nodes = [
        'n2b1',
        'n2b2',
        'n2b3',
        'n2b4',
        'n3b1',
        'n3b2',
        'n4b1',
        'n4b2',
    ]

    for device in devices:
        print(device)
        nodes_snr_before = []
        nodes_snr_after = []
        for node in nodes:

            clean_acts = torch.load(
                f"run/mnist.four0123.train.noaddnoise.bnormnolast.u3cu3_0."
                f"{node}.default/activations/mnist.four0123.eval.tq.300_s18400.pt")

            noisy_acts = torch.load(
                f"run/mnist.four0123.train.noaddnoise.bnormnolast.u3cu3_0."
                f"{node}.default/activations/mnist.four0123.eval.{device}.real"
                f".opt2.noancilla.300_s18400.pt")

            # logger.info(f'{device}, {node}')
            snr_befores = []
            snr_afters = []
            for k, (clean_act, noisy_act) in enumerate(zip(clean_acts, noisy_acts)):
                if k == len(clean_acts) - 1:
                    continue
                # logger.info(f"Node {k}")
                for stage in ['x_before_add_noise',
                              'x_before_act_quant',
                              ]:
                    diff = noisy_act[stage] - clean_act[stage]

                    # Compute SNR
                    snr_all = clean_act[stage].square().mean() / diff.square().mean()
                    if stage == 'x_before_add_noise':
                        snr_befores.append(snr_all.item())
                    else:
                        snr_afters.append(snr_all.item())
            # logger.info(f"{}{}")
            nodes_snr_before.append(f"{np.array(snr_befores).mean()}")
            nodes_snr_after.append(f"{np.array(snr_afters).mean()}")
        print('\n'.join(nodes_snr_before), '\n')
        print('\n'.join(nodes_snr_after))








