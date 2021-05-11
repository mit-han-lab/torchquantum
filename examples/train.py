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

    # for split in dataset:
    #     sampler = torch.utils.data.distributed.DistributedSampler(
    #         dataset[split],
    #         num_replicas=dist.size(),
    #         rank=dist.rank(),
    #         shuffle=(split == 'train'))
    #     dataflow[split] = torch.utils.data.DataLoader(
    #         dataset[split],
    #         batch_size=configs.run.bsz // dist.size(),
    #         sampler=sampler,
    #         num_workers=configs.run.workers_per_gpu,
    #         pin_memory=True)

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.run.bsz,
            sampler=sampler,
            num_workers=configs.run.workers_per_gpu,
            pin_memory=True)

    model = builder.make_model()

    state_dict = {}
    solution = None
    score = None
    if configs.ckpt.load_ckpt:
        logger.warning('Loading checkpoint!')
        state_dict = io.load(
            os.path.join(args.ckpt_dir, configs.ckpt.name),
            map_location='cpu')
        if getattr(state_dict, 'model_arch', None) is not None:
            model_load = state_dict['model_arch']
            for module_load, module in zip(model_load.modules(), model.modules()):
                if isinstance(module, tq.RandomLayer):
                    # random layer, need to restore the architecture
                    module.rebuild_random_layer_from_op_list(
                        n_ops_in=module_load.n_ops,
                        wires_in=module_load.wires,
                        op_list_in=module_load.op_list,
                    )

        if not configs.ckpt.weight_from_scratch:
            model.load_state_dict(state_dict['model'], strict=False)
        else:
            logger.warning(f"DO NOT load weight, train weights from scratch!")

        if 'solution' in state_dict.keys():
            solution = state_dict['solution']
            logger.info(f"Loading the solution {solution}")
            logger.info(f"Original score: {state_dict['score']}")
            model.set_sample_arch(solution['arch'])
            score = state_dict['score']

        if 'v_c_reg_mapping' in state_dict.keys():
            try:
                model.measure.set_v_c_reg_mapping(
                    state_dict['v_c_reg_mapping'])
            except AttributeError:
                logger.warning(f"Cannot set v_c_reg_mapping.")

        if configs.model.load_op_list:
            assert state_dict['q_layer_op_list'] is not None
            logger.warning(f"Loading the op_list, will replace the q_layer in "
                           f"the original model!")
            q_layer = build_module_from_op_list(state_dict['q_layer_op_list'])
            model.q_layer = q_layer

    if configs.model.transpile_before_run:
        # transpile the q_layer
        logger.warning(f"Transpile the q_layer to basis gate set before "
                       f"training, will replace the q_layer!")
        processor = builder.make_qiskit_processor()

        circ = tq2qiskit(model.q_device, model.q_layer)

        """
        add measure because the transpile process may permute the wires, 
        so we need to get the final q reg to c reg mapping 
        """
        circ.measure(list(range(model.q_device.n_wires)),
                     list(range(model.q_device.n_wires)))

        logger.info("Transpiling circuit...")

        if solution is not None:
            processor.set_layout(solution['layout'])
            logger.warning(f"Set layout {solution['layout']} for transpile!")

        circ_transpiled = processor.transpile(circs=circ)
        q_layer = qiskit2tq(circ=circ_transpiled)

        model.measure.set_v_c_reg_mapping(
            get_v_c_reg_mapping(circ_transpiled))
        model.q_layer = q_layer

        if configs.trainer.add_noise:
            # noise-aware training
            noise_model_tq = tq.NoiseModelTQ(
                backend_name=configs.qiskit.backend_name,
                noise_total_prob=configs.trainer.noise_total_prob,
            )
            noise_model_tq.is_add_noise = True
            noise_model_tq.v_c_reg_mapping = get_v_c_reg_mapping(
                circ_transpiled)
            noise_model_tq.p_c_reg_mapping = get_p_c_reg_mapping(
                circ_transpiled)
            noise_model_tq.p_v_reg_mapping = get_p_v_reg_mapping(
                circ_transpiled)
            model.set_noise_model_tq(noise_model_tq)

    if getattr(configs.model.arch, 'sample_arch', None) is not None and \
            not configs.model.transpile_before_run:
        sample_arch = configs.model.arch.sample_arch
        logger.warning(f"Setting sample arch {sample_arch} from config file!")
        if isinstance(sample_arch, str):
            # this is the name of arch
            sample_arch = get_named_sample_arch(model.arch_space, sample_arch)
            logger.warning(f"Decoded sample arch: {sample_arch}")
        model.set_sample_arch(sample_arch)

    if configs.trainer.name == 'pruning_trainer':
        """
        in pruning, convert the super layers to module list, otherwise the 
        pruning ratio is difficulty to set
        """
        logger.warning(f"Convert sampled layer to module list layer!")
        model.q_layer = build_module_from_op_list(
            build_module_op_list(model.q_layer)
        )

    model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model.cuda(),
    #     device_ids=[dist.local_rank()],
    #     find_unused_parameters=True)
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

    # trainer state_dict will be loaded in a callback
    callbacks = builder.make_callbacks(dataflow, state_dict)

    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.run.n_epochs,
        callbacks=callbacks)


if __name__ == '__main__':
    main()
