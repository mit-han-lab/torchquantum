from torchpack.utils.logging import logger
import argparse
from matplotlib.ticker import FormatStrFormatter

import torch

import matplotlib.pyplot as plt
import os
import numpy as np


import numpy as np
from scipy.ndimage.filters import gaussian_filter

def kl(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    cond = (p != 0)

    return np.sum(np.where(cond, p * np.log(p / q), 0))

def smoothed_hist_kl_distance(a, b, nbins=40, sigma=1):

    mini = min(a.min(), b.min())
    maxi = max(a.max(), b.max())

    ahist, bhist = (np.histogram(a, bins=nbins,
                                 range=(mini, maxi),
                                 )[0],
                    np.histogram(b, bins=nbins,
                                 range=(mini, maxi),
                                 )[0])


    ahist = ahist / ahist.sum()
    bhist = bhist / bhist.sum()
    # asmooth = ahist
    # bsmooth = bhist

    asmooth, bsmooth = (gaussian_filter(ahist, sigma),
                        gaussian_filter(bhist, sigma))


    return kl(asmooth, bsmooth)

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

    act_quant = '.act_quant'

    if args.pdb:
        import pdb
        pdb.set_trace()

    clean_acts = torch.load(args.clean)

    # separators = np.linspace(-1, 1, 6)
    separators = [1]

    for k, clean_act in enumerate(clean_acts):
        if k == len(clean_acts) - 1:
            continue
        logger.info(f"Node {k}")
        for stage in ['x_before_add_noise',
                      'x_before_act_quant',
                      # 'x_all_norm',
                      # 'x_batch_norm',
                      # 'x_layer_norm'
                      ]:
            logger.info(f"{stage}")
            logger.info(f"clean mean {clean_act[stage].mean(0)}, "
                        f"std {clean_act[stage].std(0)}")

