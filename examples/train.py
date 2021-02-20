import argparse
import sys
import pdb
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import (InferenceRunner, MeanAbsoluteError, MinSaver,
                                 Saver, SaverRestore, CategoricalAccuracy)
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.trainers import QTrainer


def main() -> None:
    # dist.init()
    torch.backends.cudnn.benchmark = True
    # torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.debug.pdb:
        pdb.set_trace()

    if configs.debug.set_seed:
        torch.manual_seed(configs.debug.seed)
        np.random.seed(configs.debug.seed)

    if configs.run.device == 'gpu':
        device = torch.device('cuda')
    elif configs.run.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(configs.run.device)

    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    dataset = builder.make_dataset()
    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.run.bsz,
            sampler=sampler,
            num_workers=configs.run.workers_per_gpu,
            pin_memory=True)
    model = builder.make_model()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Size: {total_params}')

    # logger.info(f'Model MACs: {profile_macs(model, inputs)}')

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = QTrainer(model=model,
                       criterion=criterion,
                       optimizer=optimizer,
                       scheduler=scheduler)
    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.run.n_epochs,
        callbacks=[
            # SaverRestore(),
            InferenceRunner(
                dataflow=dataflow['valid'],
                callbacks=[CategoricalAccuracy(name='error/valid')]),
            InferenceRunner(
                dataflow=dataflow['test'],
                callbacks=[CategoricalAccuracy(name='error/test')]),
            MinSaver('error/valid'),
            # Saver(),
        ])


if __name__ == '__main__':
    main()
