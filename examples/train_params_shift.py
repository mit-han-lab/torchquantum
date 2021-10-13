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

from torchpack.utils import io
# from torchpack import distributed as dist
from torchpack.environ import set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from core import builder
from torchquantum.plugins import tq2qiskit, qiskit2tq
from torchquantum.utils import (build_module_from_op_list,
                                build_module_op_list,
                                get_v_c_reg_mapping,
                                get_p_c_reg_mapping,
                                get_p_v_reg_mapping,
                                get_cared_configs)
from torchquantum.super_utils import get_named_sample_arch
from tensorflow.python.summary.summary_iterator import summary_iterator


def main() -> None:
    # dist.init()
    torch.backends.cudnn.benchmark = True
    # torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    parser.add_argument('--print-configs', action='store_true',
                        help='print ALL configs')
    parser.add_argument('--loadGlobalStep', type=str, default=None)
    parser.add_argument('--loadDir', metavar='DIR', help='load tensorboard directory')
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

    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)

    # set the run dir according to config file's name
    args.run_dir = 'runs/' + args.config.replace('/', '.').replace(
        'examples.', '').replace('.yml', '').replace('configs.', '')
    set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))

    if args.print_configs:
        print_conf = configs
    else:
        print_conf = get_cared_configs(configs, 'train')

    logger.info(f'Training started: "{args.run_dir}".' + '\n' +
                f'{print_conf}')

    dataset = builder.make_dataset()
    dataflow = dict()

    for split in dataset:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset[split])
            batch_size = configs.run.bsz
        else:
            # for valid and test, use SequentialSampler to make the train.py
            # and eval.py results consistent
            sampler = torch.utils.data.SequentialSampler(dataset[split])
            batch_size = getattr(configs.run, 'eval_bsz', configs.run.bsz)

        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=batch_size,
            sampler=sampler,
            num_workers=configs.run.workers_per_gpu,
            pin_memory=True)

    model = builder.make_model()

    state_dict = {}
    solution = None
    score = None

    if configs.qiskit.use_qiskit_train or configs.qiskit.use_qiskit_valid:
        from torchquantum.plugins import QiskitProcessor
        processor = QiskitProcessor(use_real_qc=configs.qiskit.use_real_qc, n_shots=configs.qiskit.n_shots, backend_name=configs.qiskit.backend_name)
        model.set_qiskit_processor(processor)

    model.to(device)
    
    if getattr(model, 'sample_arch', None) is not None and \
            not configs.model.transpile_before_run and \
            not configs.trainer.name == 'pruning_trainer':
        n_params = model.count_sample_params()
        logger.info(f"Number of sampled params: {n_params}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Size: {total_params}')

    # logger.info(f'Model MACs: {profile_macs(model, inputs)}')

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)
    trainer = builder.make_trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )
    trainer.solution = solution
    trainer.score = score

    callbacks = builder.make_callbacks(dataflow, state_dict)
    
    if args.loadGlobalStep is not None:
        path = args.loadDir
        stopStep = int(args.loadGlobalStep)
        grad_dict = {}
        for i in range(1, stopStep + 1):
            grad_dict[i] = {}
        for summary in summary_iterator(path):
            if (len(summary.summary.value)) == 0:
                continue
            tag = summary.summary.value[0].tag
            value = summary.summary.value[0].simple_value
            if tag[:10] == 'grad/grad_' and 1 <= summary.step <= stopStep:
                param_id = int(tag[10:])
                grad_dict[summary.step][param_id] = value
                # logger.info('grad_{0}={1}, step={2}'.format(param_id, value, summary.step))
        trainer.load_grad(stopStep, grad_dict)

    trainer.set_use_qiskit(configs)
    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.run.n_epochs,
        callbacks=callbacks)


if __name__ == '__main__':
    main()
