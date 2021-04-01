import argparse
import os
import pdb
import torch
import torch.backends.cudnn
import tqdm
import torch.nn.functional as F
import torchquantum as tq

from torchpack.utils import io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from core import builder
from torchquantum.utils import (legalize_unitary, build_module_from_op_list,
                                get_q_c_reg_mapping, get_cared_configs,
                                get_success_rate)
from torchquantum.plugins import tq2qiskit, qiskit2tq, tq2qiskit_parameterized
from torchquantum.super_utils import get_named_sample_arch


def log_acc(output_all, target_all, k=1):
    _, indices = output_all.topk(k, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Loss: {loss}")


def main() -> None:
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    parser.add_argument('--print-configs', action='store_true',
                        help='print ALL configs')
    parser.add_argument('--jobs', type=int, default=None,
                        help='max parallel job on qiskit')
    args, opts = parser.parse_known_args()

    configs.load(os.path.join(args.run_dir, 'metainfo', 'configs.yaml'))
    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.debug.pdb or args.pdb:
        pdb.set_trace()

    configs.verbose = args.verbose

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.jobs is not None:
        configs.qiskit.max_jobs = args.jobs

    if configs.run.device == 'gpu':
        device = torch.device('cuda')
    elif configs.run.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(configs.run.device)

    if args.print_configs:
        print_conf = configs
    else:
        print_conf = get_cared_configs(configs, 'eval')

    logger.info(f'Evaluation started: "{args.run_dir}".' + '\n' +
                f'{print_conf}')

    # if configs.qiskit.use_qiskit:
    #     IBMQ.load_account()
    #     if configs.run.bsz == 'qiskit_max':
    #         configs.run.bsz = IBMQ.get_provider(hub='ibm-q').get_backend(
    #             configs.qiskit.backend_name).configuration().max_experiments

    dataset = builder.make_dataset()
    sampler = torch.utils.data.SequentialSampler(dataset[
                                                     configs.dataset.split])
    dataflow = torch.utils.data.DataLoader(
        dataset[configs.dataset.split],
        sampler=sampler,
        batch_size=configs.run.bsz,
        num_workers=configs.run.workers_per_gpu,
        pin_memory=True)

    state_dict = io.load(
        os.path.join(args.run_dir, configs.ckpt.name),
        map_location='cpu')
    model_load = state_dict['model_arch']
    model = builder.make_model()
    for module_load, module in zip(model_load.modules(), model.modules()):
        if isinstance(module, tq.RandomLayer):
            # random layer, need to restore the architecture
            module.rebuild_random_layer_from_op_list(
                n_ops_in=module_load.n_ops,
                wires_in=module_load.wires,
                op_list_in=module_load.op_list,
            )

    model.load_state_dict(state_dict['model'], strict=False)

    if 'solution' in state_dict.keys():
        solution = state_dict['solution']
        logger.info(f"Evaluate the solution {solution}")
        logger.info(f"Original score: {state_dict['score']}")
        model.set_sample_arch(solution['arch'])

    if 'q_c_reg_mapping' in state_dict.keys():
        try:
            model.measure.set_q_c_reg_mapping(state_dict['q_c_reg_mapping'])
        except AttributeError:
            logger.warning(f"Cannot set q_c_reg_mapping.")

    if configs.model.load_op_list:
        assert state_dict['q_layer_op_list'] is not None
        logger.warning(f"Loading the op_list, will replace the q_layer in "
                       f"the original model!")
        q_layer = build_module_from_op_list(
            op_list=state_dict['q_layer_op_list'],
            remove_ops=configs.prune.eval.remove_ops,
            thres=configs.prune.eval.remove_ops_thres)
        model.q_layer = q_layer

    if configs.model.transpile_before_run:
        # transpile the q_layer
        logger.warning(f"Transpile the q_layer to basis gate set before "
                       f"evaluation, will replace the q_layer!")
        processor = builder.make_qiskit_processor()

        circ = tq2qiskit(model.q_device, model.q_layer)

        """
        add measure because the transpile process may permute the wires, 
        so we need to get the final q reg to c reg mapping 
        """
        circ.measure(list(range(model.q_device.n_wires)),
                     list(range(model.q_device.n_wires)))

        logger.info("Transpiling circuit...")
        circ_transpiled = processor.transpile(circs=circ)
        q_layer = qiskit2tq(circ=circ_transpiled)

        model.measure.set_q_c_reg_mapping(
            get_q_c_reg_mapping(circ_transpiled))
        model.q_layer = q_layer

    if configs.legalization.legalize:
        legalize_unitary(model)
    model.to(device)
    model.eval()

    if configs.qiskit.use_qiskit:
        qiskit_processor = builder.make_qiskit_processor()
        if configs.qiskit.initial_layout is not None:
            layout = configs.qiskit.initial_layout
            logger.warning(f"Use layout {layout} from config file")
        elif 'solution' in state_dict.keys():
            layout = state_dict['solution']['layout']
            logger.warning(f"Use layout {layout} from checkpoint file")
        else:
            layout = None
            logger.warning(f"No specified layout")
        qiskit_processor.set_layout(layout)
        model.set_qiskit_processor(qiskit_processor)

    if getattr(configs.model.arch, 'sample_arch', None) is not None:
        sample_arch = configs.model.arch.sample_arch
        logger.warning(f"Setting sample arch {sample_arch}")
        if isinstance(sample_arch, str):
            # this is the name of arch
            sample_arch = get_named_sample_arch(model.arch_space, sample_arch)
            logger.warning(f"Decoded sample arch: {sample_arch}")
        model.set_sample_arch(sample_arch)

    if configs.qiskit.est_success_rate:
        circ_parameterized, params = tq2qiskit_parameterized(
            model.q_device, model.encoder.func_list)
        circ_fixed = tq2qiskit(model.q_device, model.q_layer)
        circ = circ_parameterized + circ_fixed
        transpiled_circ = model.qiskit_processor.transpile(circ)

        success_rate = get_success_rate(
            model.qiskit_processor.properties,
            transpiled_circ)
        logger.info(f"Success_rate: {success_rate}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Size: {total_params}')

    with torch.no_grad():
        target_all = None
        output_all = None
        for feed_dict in tqdm.tqdm(dataflow):
            if configs.run.device == 'gpu':
                inputs = feed_dict['image'].cuda(non_blocking=True)
                targets = feed_dict['digit'].cuda(non_blocking=True)
            else:
                inputs = feed_dict['image']
                targets = feed_dict['digit']

            if configs.qiskit.use_qiskit:
                outputs = model.forward_qiskit(inputs, verbose=configs.verbose)
            else:
                outputs = model(inputs, verbose=configs.verbose)

            if target_all is None:
                target_all = targets
                output_all = outputs
            else:
                target_all = torch.cat([target_all, targets], dim=0)
                output_all = torch.cat([output_all, outputs], dim=0)
            # if configs.verbose:
            #     logger.info(f"Measured log_softmax: {outputs}")
            log_acc(output_all, target_all)

    logger.info("Final:")
    log_acc(output_all, target_all)


if __name__ == '__main__':
    main()
