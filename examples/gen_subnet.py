import argparse
import pdb
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data

from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from core import builder
from torchquantum.utils import get_cared_configs
from torchquantum.super_utils import get_named_sample_arch, ArchSampler


def main() -> None:
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--ckpt-dir', metavar='DIR', help='run directory')
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    parser.add_argument('--print-configs', action='store_true',
                        help='print ALL configs')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.debug.pdb or args.pdb:
        pdb.set_trace()

    if configs.debug.set_seed:
        torch.manual_seed(configs.debug.seed)
        np.random.seed(configs.debug.seed)

    if args.print_configs:
        print_conf = configs
    else:
        print_conf = get_cared_configs(configs, 'train')

    logger.info(f"Generate subnet started: \n {print_conf}")

    model = builder.make_model()

    sampler = ArchSampler(
        model,
        strategy=configs.model.sampler.strategy,
        n_layers_per_block=configs.model.arch.n_layers_per_block)

    arch_all = []
    for _ in range(100):
        arch = sampler.get_random_sample_arch()
        arch_all.append(arch)

    arch_all.sort(key=lambda x: x[-1])

    for arch in arch_all:
        print(f"blk {arch[-1]} arch: {arch}")


if __name__ == '__main__':
    main()
