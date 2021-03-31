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
from torchquantum.utils import (legalize_unitary, build_module_op_list,
                                get_cared_configs)
from torchquantum.plugins import tq2qiskit, tq2qiskit_parameterized
from qiskit import IBMQ
from core.tools import EvolutionEngine
from qiskit.providers.aer.noise.device.parameters import gate_error_values
from torch.utils.tensorboard import SummaryWriter


def get_success_rate(properties, transpiled_circ):
    # estimate the success rate according to the error rates of single and
    # two-qubit gates in transpiled circuits

    gate_errors = gate_error_values(properties)
    # construct the error dict
    gate_error_dict = {}
    for gate_error in gate_errors:
        if gate_error[0] not in gate_error_dict.keys():
            gate_error_dict[gate_error[0]] = {tuple(gate_error[1]):
                                              gate_error[2]}
        else:
            gate_error_dict[gate_error[0]][tuple(gate_error[1])] = \
                gate_error[2]

    success_rate = 1
    for gate in transpiled_circ.data:
        gate_success_rate = 1 - gate_error_dict[gate[0].name][tuple(
            map(lambda x: x.index, gate[1])
        )]
        success_rate *= gate_success_rate

    return success_rate


def evaluate_all(model, dataflow, solutions, writer=None, iter_n=None,
                 population_size=None):
    scores = []

    best_solution_accuracy = 0
    best_solution_loss = 0
    best_solution_success_rate = 0
    best_solution_score = 999999

    for i, solution in tqdm.tqdm(enumerate(solutions)):
        if model.qiskit_processor is not None:
            model.qiskit_processor.set_layout(solution['layout'])
        model.set_sample_arch(solution['arch'])
        with torch.no_grad():
            target_all = None
            output_all = None
            for feed_dict in dataflow:
                if configs.run.device == 'gpu':
                    inputs = feed_dict['image'].cuda(non_blocking=True)
                    targets = feed_dict['digit'].cuda(non_blocking=True)
                else:
                    inputs = feed_dict['image']
                    targets = feed_dict['digit']
                if configs.qiskit.use_qiskit:
                    outputs = model.forward_qiskit(inputs)
                else:
                    outputs = model.forward(inputs)

                if target_all is None:
                    target_all = targets
                    output_all = outputs
                else:
                    target_all = torch.cat([target_all, targets], dim=0)
                    output_all = torch.cat([output_all, outputs], dim=0)

        k = 1
        _, indices = output_all.topk(k, dim=1)
        masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
        size = target_all.shape[0]
        corrects = masks.sum().item()
        accuracy = corrects / size
        loss = F.nll_loss(output_all, target_all).item()

        if configs.es.est_success_rate:
            circ_parameterized, params = tq2qiskit_parameterized(
                model.q_device, model.encoder.func_list)
            circ_fixed = tq2qiskit(model.q_device, model.q_layer)
            circ = circ_parameterized + circ_fixed
            transpiled_circ = model.qiskit_processor.transpile(circ)

            success_rate = get_success_rate(
                model.qiskit_processor.properties,
                transpiled_circ)
        else:
            success_rate = 1
        score = loss / success_rate
        scores.append(score)
        logger.info(f"Accuracy: {accuracy:.5f}, Loss: {loss:.5f}, "
                    f"Success Rate: {success_rate: .5f}, Score: {score:.5f}")

        if score < best_solution_score:
            best_solution_accuracy = accuracy
            best_solution_success_rate = success_rate
            best_solution_loss = loss
            best_solution_score = score

        logger.info(f"Best of iteration: "
                    f"Accuracy: {best_solution_accuracy:.5f}, "
                    f"Loss: {best_solution_loss:.5f}, "
                    f"Success Rate: {best_solution_success_rate: .5f}, "
                    f"Score: {best_solution_score:.5f}")

        if population_size is not None and writer is not None and \
                population_size is not None:
            writer.add_scalar('es/accuracy', accuracy,
                              iter_n * population_size + i)
            writer.add_scalar('es/loss', loss, iter_n * population_size + i)
            writer.add_scalar('es/success_rate', success_rate,
                              iter_n * population_size + i)
            writer.add_scalar('es/score', score, iter_n * population_size + i)

    return scores, best_solution_accuracy, best_solution_loss, \
        best_solution_success_rate, best_solution_score


def main() -> None:
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    parser.add_argument('--jobs', type=int, default=None,
                        help='max parallel job on qiskit')
    parser.add_argument('--print-configs', action='store_true',
                        help='print ALL configs')
    args, opts = parser.parse_known_args()

    configs.load(os.path.join(args.run_dir, 'metainfo', 'configs.yaml'))
    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.debug.pdb or args.pdb:
        pdb.set_trace()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.jobs is not None:
        configs.qiskit.max_jobs = args.jobs

    # if use qiskit, then not need to estimate success rate
    assert not (configs.es.est_success_rate and configs.qiskit.use_qiskit)

    if configs.run.device == 'gpu':
        device = torch.device('cuda')
    elif configs.run.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(configs.run.device)

    if args.print_configs:
        print_conf = configs
    else:
        print_conf = get_cared_configs(configs, 'es')

    logger.info(f'Evolutionary Search started: "{args.run_dir}".' + '\n' +
                f'{print_conf}')

    if configs.qiskit.use_qiskit:
        IBMQ.load_account()
        if configs.run.bsz == 'qiskit_max':
            configs.run.bsz = IBMQ.get_provider(hub='ibm-q').get_backend(
                configs.qiskit.backend_name).configuration().max_experiments

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
        map_location=device)

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

    if configs.legalization.legalize:
        legalize_unitary(model)
    model.to(device)
    model.eval()

    es_dir = 'es_runs/' + args.config.replace('/', '.').replace(
        'examples.', '').replace('.yml', '').replace('configs.', '')
    io.save(os.path.join(es_dir, 'metainfo/configs.yaml'), configs.dict())

    writer = SummaryWriter(os.path.normpath(os.path.join(es_dir,
                                                         'tb')))

    if configs.qiskit.use_qiskit or configs.es.est_success_rate:
        IBMQ.load_account()
        properties = IBMQ.get_provider(hub='ibm-q').get_backend(
            configs.qiskit.backend_name).properties()
        n_available_wires = len(properties.qubits)
        qiskit_processor = builder.make_qiskit_processor()
        model.set_qiskit_processor(qiskit_processor)
    else:
        n_available_wires = model.q_device.n_wires

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Size: {total_params}')

    es_engine = EvolutionEngine(
        population_size=configs.es.population_size,
        parent_size=configs.es.parent_size,
        mutation_size=configs.es.mutation_size,
        mutation_prob=configs.es.mutation_prob,
        crossover_size=configs.es.crossover_size,
        n_wires=model.q_device.n_wires,
        n_available_wires=n_available_wires,
        arch_space=model.arch_space,
    )

    logger.info(f"Start Evolution Search")
    for k in range(configs.es.n_iterations):
        logger.info(f"ES iteration {k}:")
        solutions = es_engine.ask()
        scores, best_solution_accuracy, best_solution_loss, \
            best_solution_success_rate, best_solution_score \
            = evaluate_all(model, dataflow, solutions, writer, k,
                           configs.es.population_size)
        es_engine.tell(scores)
        logger.info(f"Best solution: {es_engine.best_solution}")
        logger.info(f"Best score: {es_engine.best_score}")

        assert best_solution_score == es_engine.best_score
        writer.add_text('es/best_solution_arch',
                        str(es_engine.best_solution), k)
        writer.add_scalar('es/best_solution_accuracy',
                          best_solution_accuracy, k)
        writer.add_scalar('es/best_solution_loss', best_solution_loss, k)
        writer.add_scalar('es/best_solution_success_rate',
                          best_solution_success_rate, k)
        writer.add_scalar('es/best_solution_score', es_engine.best_score, k)

        # store the model and solution after every iteration
        state_dict = dict()
        state_dict['model_arch'] = model
        state_dict['model'] = model.state_dict()
        state_dict['solution'] = es_engine.best_solution
        state_dict['score'] = es_engine.best_score

        state_dict['encoder_func_list'] = model.encoder.func_list
        state_dict['q_layer_op_list'] = build_module_op_list(model.q_layer)
        io.save(os.path.join(es_dir, 'checkpoints/best_solution.pt'),
                state_dict)

    logger.info(f"\n Best solution evaluation on tq:")
    # eval the best solution and save the model
    evaluate_all(model, dataflow, [es_engine.best_solution])

    # eval with the noise model
    if configs.es.eval.use_noise_model:
        logger.info(f"\n Best solution evaluation with noise model "
                    f"of {configs.qiskit.noise_model_name}:")
        configs.qiskit.use_qiskit = True
        evaluate_all(model, dataflow, [es_engine.best_solution])

    # eval on real QC
    if configs.es.eval.use_real_qc:
        logger.info(f"\n Best solution evaluation on real "
                    f"QC {configs.qiskit.backend_name}:")

        # need reset some parameters
        configs.qiskit.use_qiskit = True
        model.qiskit_processor.use_real_qc = True
        model.qiskit_processor.noise_model_name = None
        model.qiskit_processor.qiskit_init()

        # if configs.es.eval.bsz == 'qiskit_max':
        #     configs.run.bsz = \
        #         model.qiskit_processor.backend.configuration().max_experiments
        # else:
        configs.run.bsz = configs.es.eval.bsz

        configs.dataset.n_test_samples = configs.es.eval.n_test_samples
        dataset = builder.make_dataset()
        sampler = torch.utils.data.SequentialSampler(dataset['test'])
        dataflow = torch.utils.data.DataLoader(
            dataset['test'],
            sampler=sampler,
            batch_size=configs.run.bsz,
            num_workers=configs.run.workers_per_gpu,
            pin_memory=True)

        evaluate_all(model, dataflow, [es_engine.best_solution])


if __name__ == '__main__':
    main()
