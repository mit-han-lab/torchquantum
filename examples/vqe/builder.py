import torch
import torch.nn as nn

from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler
from torchpack.callbacks import (InferenceRunner, MeanAbsoluteError,
                                 MaxSaver, MinSaver,
                                 Saver, SaverRestore, CategoricalAccuracy)
from examples.gradient_pruning.callbacks import LegalInferenceRunner, SubnetInferenceRunner, \
    NLLError, TrainerRestore, AddNoiseInferenceRunner, GradRestore
from torchquantum.plugins import QiskitProcessor
from torchquantum.vqe_utils import parse_hamiltonian_file
from torchquantum.noise_model import *

__all__ = ['make_optimizer',
    'make_scheduler', 'make_callbacks', 'make_qiskit_processor',
    'make_noise_model_tq'
]

def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=configs.optimizer.lr,
            momentum=configs.optimizer.momentum,
            weight_decay=configs.optimizer.weight_decay,
            nesterov=configs.optimizer.nesterov)
    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adam_group':
        optimizer = torch.optim.Adam(
            [param for name, param in model.named_parameters() if
                'lambda' not in name],
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
        optimizer.add_param_group({
            'params': [param for name, param in model.named_parameters() if
                       'lambda' in name],
            'lr': -configs.optimizer.lambda_lr,
        })
    elif configs.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)

    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'constant':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.run.n_epochs)
    elif configs.scheduler.name == 'cosine_warm':
        from .schedulers import CosineAnnealingWarmupRestarts
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.optimizer.min_lr,
            warmup_steps=configs.run.n_warm_epochs,
        )
    else:
        raise NotImplementedError(configs.scheduler.name)

    return scheduler


def make_trainer(model: nn.Module,
                 criterion: nn.Module,
                 optimizer: Optimizer,
                 scheduler: Scheduler
                 ):
    if configs.trainer.name == 'params_shift_trainer':
        from examples.gradient_pruning.trainers import ParamsShiftTrainer
        trainer = ParamsShiftTrainer(model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=scheduler)
    elif configs.trainer.name == 'q_trainer':
        from examples.gradient_pruning.trainers import QTrainer
        trainer = QTrainer(model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=scheduler)
    else:
        raise NotImplementedError(configs.trainer.name)

    return trainer


def get_subcallbacks(config):
    subcallbacks = []
    for subcallback in config:
        if subcallback['metrics'] == 'CategoricalAccuracy':
            subcallbacks.append(
                CategoricalAccuracy(name=subcallback['name'])
            )
        elif subcallback['metrics'] == 'MeanAbsoluteError':
            subcallbacks.append(
                MeanAbsoluteError(name=subcallback['name'])
            )
        elif subcallback['metrics'] == 'NLLError':
            subcallbacks.append(
                NLLError(name=subcallback['name'])
            )
        elif subcallback['metrics'] == 'MinError':
            from examples.gradient_pruning.callbacks import MinError
            subcallbacks.append(
                MinError(name=subcallback['name'])
            )
        else:
            raise NotImplementedError(subcallback['metrics'])
    return subcallbacks


def make_callbacks(dataflow, state=None):
    callbacks = []
    for config in configs['callbacks']:
        if config['callback'] == 'InferenceRunner':
            callback = InferenceRunner(
                dataflow=dataflow[config['split']],
                callbacks=get_subcallbacks(config['subcallbacks'])
            )
        elif config['callback'] == 'LegalInferenceRunner':
            callback = LegalInferenceRunner(
                dataflow=dataflow[config['split']],
                callbacks=get_subcallbacks(config['subcallbacks'])
            )
        elif config['callback'] == 'SubnetInferenceRunner':
            callback = SubnetInferenceRunner(
                dataflow=dataflow[config['split']],
                callbacks=get_subcallbacks(config['subcallbacks']),
                subnet=config['subnet']
            )
        elif config['callback'] == 'AddNoiseInferenceRunner':
            callback = AddNoiseInferenceRunner(
                dataflow=dataflow[config['split']],
                callbacks=get_subcallbacks(config['subcallbacks']),
                noise_total_prob=config['noise_total_prob']
            )
        elif config['callback'] == 'SaverRestore':
            callback = SaverRestore()
        elif config['callback'] == 'Saver':
            callback = Saver(max_to_keep=config['max_to_keep'])
        elif config['callback'] == 'MaxSaver':
            callback = MaxSaver(config['name'])
        elif config['callback'] == 'MinSaver':
            callback = MinSaver(config['name'])
        elif config['callback'] == 'GradRestore':
            callback = GradRestore()
        else:
            raise NotImplementedError(config['callback'])
        callbacks.append(callback)

    if configs.ckpt.load_trainer:
        assert state is not None
        callback = TrainerRestore(state)
        callbacks.append(callback)

    return callbacks


def make_qiskit_processor():
    processor = QiskitProcessor(
        use_real_qc=configs.qiskit.use_real_qc,
        backend_name=configs.qiskit.backend_name,
        noise_model_name=configs.qiskit.noise_model_name,
        coupling_map_name=getattr(configs.qiskit, 'coupling_map_name',
                                  configs.qiskit.noise_model_name),
        basis_gates_name=configs.qiskit.noise_model_name,
        n_shots=configs.qiskit.n_shots,
        initial_layout=configs.qiskit.initial_layout,
        seed_transpiler=configs.qiskit.seed_transpiler,
        seed_simulator=configs.qiskit.seed_simulator,
        optimization_level=configs.qiskit.optimization_level,
        max_jobs=configs.qiskit.max_jobs,
        remove_ops=configs.prune.eval.remove_ops,
        remove_ops_thres=configs.prune.eval.remove_ops_thres,
        transpile_with_ancilla=getattr(configs.qiskit, 'transpile_with_ancilla', True),
        hub=getattr(configs.qiskit, 'hub', None),
        layout_method=getattr(configs.qiskit, 'layout_method', None),
        routing_method=getattr(configs.qiskit, 'routing_method', None),
    )
    return processor


def make_noise_model_tq():
    if configs.trainer.noise_model_tq_name == 'from_qiskit_read':
        noise_model_tq = NoiseModelTQ(
            noise_model_name=configs.qiskit.noise_model_name,
            n_epochs=configs.run.n_epochs,
            noise_total_prob=getattr(configs.trainer, 'noise_total_prob',
                                     None),
            ignored_ops=configs.trainer.ignored_noise_ops,
            prob_schedule=getattr(configs.trainer, 'noise_prob_schedule',
                                  None),
            prob_schedule_separator=getattr(
                configs.trainer, 'noise_prob_schedule_separator', None),
            factor=getattr(configs.trainer, 'noise_factor', None),
            add_thermal=getattr(configs.trainer, 'noise_add_thermal', True)
        )
    elif configs.trainer.noise_model_tq_name == 'from_qiskit':
        noise_model_tq = NoiseModelTQQErrorOnly(
            noise_model_name=configs.qiskit.noise_model_name,
            n_epochs=configs.run.n_epochs,
            noise_total_prob=getattr(configs.trainer, 'noise_total_prob',
                                     None),
            ignored_ops=configs.trainer.ignored_noise_ops,
            prob_schedule=getattr(configs.trainer, 'noise_prob_schedule',
                                  None),
            prob_schedule_separator=getattr(
                configs.trainer, 'noise_prob_schedule_separator', None),
            factor=getattr(configs.trainer, 'noise_factor', None),
            add_thermal=getattr(configs.trainer, 'noise_add_thermal', True)
        )
    elif configs.trainer.noise_model_tq_name == 'activation':
        noise_model_tq = NoiseModelTQActivation(
            mean=configs.trainer.noise_mean,
            std=configs.trainer.noise_std,
            n_epochs=configs.run.n_epochs,
            prob_schedule=getattr(configs.trainer, 'noise_prob_schedule',
                                  None),
            prob_schedule_separator=getattr(
                configs.trainer, 'noise_prob_schedule_separator', None),
            after_norm=getattr(configs.trainer, 'noise_after_norm',
                               False),
            factor=getattr(configs.trainer, 'noise_factor', None)
        )
    elif configs.trainer.noise_model_tq_name == 'phase':
        noise_model_tq = NoiseModelTQPhase(
            mean=configs.trainer.noise_mean,
            std=configs.trainer.noise_std,
            n_epochs=configs.run.n_epochs,
            prob_schedule=getattr(configs.trainer, 'noise_prob_schedule',
                                  None),
            prob_schedule_separator=getattr(
                configs.trainer, 'noise_prob_schedule_separator', None),
            factor=getattr(configs.trainer, 'noise_factor', None)
        )
    elif configs.trainer.noise_model_tq_name == 'only_read':
        noise_model_tq = NoiseModelTQReadoutOnly(
            noise_model_name=configs.qiskit.noise_model_name,
            n_epochs=configs.run.n_epochs,
            prob_schedule=getattr(configs.trainer, 'noise_prob_schedule',
                                  None),
            prob_schedule_separator=getattr(
                configs.trainer, 'noise_prob_schedule_separator', None),
            factor=getattr(configs.trainer, 'noise_factor', None)
        )
    elif configs.trainer.noise_model_tq_name == 'activation_read':
        noise_model_tq = NoiseModelTQActivationReadout(
            noise_model_name=configs.qiskit.noise_model_name,
            mean=configs.trainer.noise_mean,
            std=configs.trainer.noise_std,
            n_epochs=configs.run.n_epochs,
            prob_schedule=getattr(configs.trainer, 'noise_prob_schedule',
                                  None),
            prob_schedule_separator=getattr(
                configs.trainer, 'noise_prob_schedule_separator', None),
            after_norm=getattr(configs.trainer, 'noise_after_norm',
                               False),
            factor=getattr(configs.trainer, 'noise_factor', None)
        )
    elif configs.trainer.noise_model_tq_name == 'phase_read':
        noise_model_tq = NoiseModelTQPhaseReadout(
            noise_model_name=configs.qiskit.noise_model_name,
            mean=configs.trainer.noise_mean,
            std=configs.trainer.noise_std,
            n_epochs=configs.run.n_epochs,
            prob_schedule=getattr(configs.trainer, 'noise_prob_schedule',
                                  None),
            prob_schedule_separator=getattr(
                configs.trainer, 'noise_prob_schedule_separator', None),
            factor=getattr(configs.trainer, 'noise_factor', None)
        )
    else:
        raise NotImplementedError(configs.trainer.noise_model_tq_name)

    return noise_model_tq
