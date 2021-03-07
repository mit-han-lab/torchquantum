import argparse
import os
import pdb

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
import tqdm
from torchpack.utils import fs, io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder


def main() -> None:
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    args, opts = parser.parse_known_args()

    configs.load(os.path.join(args.run_dir, 'metainfo', 'configs.yaml'))
    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.debug.pdb or args.pdb:
        pdb.set_trace()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if configs.run.device == 'gpu':
        device = torch.device('cuda')
    elif configs.run.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(configs.run.device)

    logger.info(f'Evaluation started: "{args.run_dir}".' + '\n' + f'{configs}')

    dataset = builder.make_dataset()
    sampler = torch.utils.data.SequentialSampler(dataset['test'])
    dataflow = torch.utils.data.DataLoader(
        dataset['test'],
        sampler=sampler,
        batch_size=configs.run.bsz,
        num_workers=configs.run.workers_per_gpu,
        pin_memory=True)
    model = builder.make_model()
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Size: {total_params}')

    # inputs = {'sizing': torch.randn(1, configs.model.in_ch).cuda(),
    #           'fom': torch.randn(1).cuda()
    #           }
    # logger.info(f'Model MACs: {profile_macs(model, inputs)}')

    state_dict = io.load(
        os.path.join(args.run_dir, 'checkpoints', 'max-acc-valid.pt'))
    model.load_state_dict(state_dict['model'])

    with torch.no_grad():
        target_all = None
        output_all = None
        for feed_dict in tqdm.tqdm(dataflow):
            inputs = dict()
            for key, value in feed_dict.items():
                if key in ['sizing']:
                    inputs[key] = value.cuda()
            targets = feed_dict['fom'].cuda(non_blocking=True)

            outputs = model(inputs)

            if target_all is None:
                target_all = targets.cpu().numpy()
                output_all = outputs.cpu().numpy()
            else:
                target_all = np.concatenate([target_all,
                                             targets.cpu().numpy()])
                output_all = np.concatenate([output_all,
                                             outputs.cpu().numpy()])



if __name__ == '__main__':
    main()