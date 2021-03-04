import argparse
import os
import pdb
import sys
import numpy as np
import torch
import torch.autograd.profiler as profiler

from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from examples.core import builder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.debug.pdb or args.pdb:
        pdb.set_trace()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if configs.debug.set_seed:
        torch.manual_seed(configs.debug.seed)
        np.random.seed(configs.debug.seed)

    if configs.run.device == 'gpu':
        device = torch.device('cuda')
    elif configs.run.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(configs.run.device)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Profiling started: "{args.run_dir}".' + '\n' + f'{configs}')

    inputs = torch.tensor(torch.rand(configs.run.bsz, 1, 28, 28),
                          device=device)

    model = builder.make_model()

    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            model(inputs)

    prof.export_chrome_trace("part1_static.json")


if __name__ == '__main__':
    main()
