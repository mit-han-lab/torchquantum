import argparse
import os
import sys
import pdb
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
import torchquantum as tq

# from torchpack import distributed as dist
from torchpack.environ import set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from examples.vqe import builder
from torchquantum.vqe_utils import parse_hamiltonian_file
from examples.vqe.q_models import QVQEModel0

def main() -> None:
    # dist.init()
    torch.backends.cudnn.benchmark = True
    # torch.cuda.set_device(dist.local_rank())

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

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if configs.debug.set_seed:
        torch.manual_seed(configs.debug.seed)
        np.random.seed(configs.debug.seed)

    if configs.run.device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            configs.run.device = 'cpu'
            device = torch.device('cpu')
    elif configs.run.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(configs.run.device)

    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)

    # set the run dir according to config file's name
    args.run_dir = 'runs/' + args.config.replace('/', '.').replace(
        'examples.', '').replace('.yml', '').replace('configs.', '')
    set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))

    configs.model.hamil_filename = 'examples/vqe/h2.txt'

    print_conf = configs

    logger.info(f'Training started: "{args.run_dir}".' + '\n' +
                f'{print_conf}')

    
    from torchquantum.datasets import VQE
    dataset = VQE(
        steps_per_epoch=configs.run.steps_per_epoch
    )
    dataflow = dict()

    for split in dataset:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset[split])
        else:
            # for valid and test, use SequentialSampler to make the train.py
            # and eval.py results consistent
            sampler = torch.utils.data.SequentialSampler(dataset[split])

        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.run.bsz,
            sampler=sampler,
            num_workers=configs.run.workers_per_gpu,
            pin_memory=True)

    model = QVQEModel0(arch=configs.model.arch, hamil_info=parse_hamiltonian_file(configs.model.hamil_filename))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Size: {total_params}')

    # make criterion
    from examples.vqe.criterions import minimize
    criterion = minimize

    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    trainer = builder.make_trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # trainer state_dict will be loaded in a callback
    callbacks = builder.make_callbacks(dataflow)
    trainer.set_use_qiskit(configs)
    if configs.qiskit.use_qiskit_train or configs.qiskit.use_qiskit_valid:
        from torchquantum.plugins import QiskitProcessor
        processor = QiskitProcessor(use_real_qc=configs.qiskit.use_real_qc, n_shots=configs.qiskit.n_shots, backend_name=configs.qiskit.backend_name)
        model.set_qiskit_processor(processor)

    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.run.n_epochs,
        callbacks=callbacks)


if __name__ == '__main__':
    main()
