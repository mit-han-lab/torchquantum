import torch
import torch.nn as nn

from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler
from torchpack.callbacks import (InferenceRunner, MeanAbsoluteError,
                                 MaxSaver, MinSaver,
                                 Saver, SaverRestore, CategoricalAccuracy)
from .callbacks import LegalInferenceRunner, SubnetInferenceRunner, \
    NLLError, TrainerRestore, AddNoiseInferenceRunner
from torchquantum.plugins import QiskitProcessor
from torchquantum.vqe_utils import parse_hamiltonian_file
from torchquantum.noise_model import *

__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler', 'make_callbacks', 'make_qiskit_processor',
    'make_noise_model_tq'
]


def make_dataset() -> Dataset:
    if configs.dataset.name == 'mnist':
        from .datasets import MNIST
        dataset = MNIST(
            root=configs.dataset.root,
            train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
            center_crop=configs.dataset.center_crop,
            resize=configs.dataset.resize,
            resize_mode=configs.dataset.resize_mode,
            binarize=configs.dataset.binarize,
            binarize_threshold=configs.dataset.binarize_threshold,
            digits_of_interest=configs.dataset.digits_of_interest,
            n_test_samples=configs.dataset.n_test_samples,
            n_valid_samples=configs.dataset.n_valid_samples,
            fashion=configs.dataset.fashion,
        )
    elif configs.dataset.name == 'layer_regression':
        from .datasets import LayerRegression
        dataset = LayerRegression()
    elif configs.dataset.name == 'vowel':
        from .datasets import Vowel
        dataset = Vowel(
            root=configs.dataset.root,
            test_ratio=configs.dataset.test_ratio,
            train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
            resize=configs.dataset.resize,
            binarize=configs.dataset.binarize,
            binarize_threshold=configs.dataset.binarize_threshold,
            digits_of_interest=configs.dataset.digits_of_interest,
        )
    elif configs.dataset.name == 'vqe':
        from .datasets import VQE
        dataset = VQE(
            steps_per_epoch=configs.run.steps_per_epoch
        )
    else:
        raise NotImplementedError(configs.dataset.name)

    return dataset


def make_model() -> nn.Module:
    if configs.model.name.startswith('t_'):
        from .models.t_models import model_dict
        model = model_dict[configs.model.name]()
    elif configs.model.name.startswith('c_'):
        from .models.c_models import model_dict
        model = model_dict[configs.model.name]()
    elif configs.model.name.startswith('q_'):
        from .models.q_models import model_dict
        model = model_dict[configs.model.name](arch=configs.model.arch)
    elif configs.model.name == 'layer_regression':
        from .models import LayerRegression
        model = LayerRegression()
    elif configs.model.name.startswith('super_'):
        from .models.super_models import model_dict
        model = model_dict[configs.model.name](arch=configs.model.arch)
    elif configs.model.name.startswith('q4digit_'):
        from .models.q4digit_models import model_dict
        model = model_dict[configs.model.name](arch=configs.model.arch)
    elif configs.model.name.startswith('super4digit_'):
        from .models.super4digit_models import model_dict
        model = model_dict[configs.model.name](arch=configs.model.arch)
    elif configs.model.name.startswith('q10digit_'):
        from .models.q10digit_models import model_dict
        model = model_dict[configs.model.name](arch=configs.model.arch)
    elif configs.model.name.startswith('vqe_'):
        from .models.q_models import model_dict
        model = model_dict[configs.model.name](
            arch=configs.model.arch,
            hamil_info=parse_hamiltonian_file(configs.model.hamil_filename))
    elif configs.model.name.startswith('supervqe'):
        from .models.super_models import model_dict
        model = model_dict[configs.model.name](
            arch=configs.model.arch,
            hamil_info=parse_hamiltonian_file(configs.model.hamil_filename))
    else:
        raise NotImplementedError(configs.model.name)

    return model


def make_criterion() -> nn.Module:
    if configs.criterion.name == 'nll':
        criterion = nn.NLLLoss()
    elif configs.criterion.name == 'mse':
        criterion = nn.MSELoss()
    elif configs.criterion.name == 'complex_mse':
        from .criterions import complex_mse
        criterion = complex_mse
    elif configs.criterion.name == 'complex_mae':
        from .criterions import complex_mae
        criterion = complex_mae
    elif configs.criterion.name == 'minimize':
        from .criterions import minimize
        criterion = minimize
    elif configs.criterion.name == 'maximize':
        from .criterions import maximize
        criterion = maximize
    else:
        raise NotImplementedError(configs.criterion.name)

    return criterion


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
            min_lr=0,
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
    if configs.trainer.name == 'q_trainer':
        from .trainers import QTrainer
        trainer = QTrainer(model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=scheduler)
    elif configs.trainer.name == 'super_q_trainer':
        from .trainers import SuperQTrainer
        trainer = SuperQTrainer(model=model,
                                criterion=criterion,
                                optimizer=optimizer,
                                scheduler=scheduler)
    elif configs.trainer.name == 'pruning_trainer':
        from .trainers import PruningTrainer
        trainer = PruningTrainer(model=model,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 scheduler=scheduler)
    elif configs.trainer.name == 'q_noise_aware_trainer':
        from .trainers import QNoiseAwareTrainer
        trainer = QNoiseAwareTrainer(model=model,
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
            from .callbacks import MinError
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
        coupling_map_name=configs.qiskit.noise_model_name,
        basis_gates_name=configs.qiskit.noise_model_name,
        n_shots=configs.qiskit.n_shots,
        initial_layout=configs.qiskit.initial_layout,
        seed_transpiler=configs.qiskit.seed_transpiler,
        seed_simulator=configs.qiskit.seed_simulator,
        optimization_level=configs.qiskit.optimization_level,
        max_jobs=configs.qiskit.max_jobs,
        remove_ops=configs.prune.eval.remove_ops,
        remove_ops_thres=configs.prune.eval.remove_ops_thres,
        transpile_with_ancilla=configs.qiskit.transpile_with_ancilla,
    )
    return processor


def make_noise_model_tq():
    if configs.trainer.noise_model_tq_name == 'from_qiskit_read':
        noise_model_tq = NoiseModelTQ(
            noise_model_name=configs.qiskit.noise_model_name,
            n_epochs=configs.run.n_epochs,
            noise_total_prob=configs.trainer.noise_total_prob,
            ignored_ops=configs.trainer.ignored_noise_ops,
            prob_schedule=getattr(configs.trainer, 'noise_prob_schedule',
                                  None),
            prob_schedule_separator=getattr(
                configs.trainer, 'noise_prob_schedule_separator', None)
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
        )
    elif configs.trainer.noise_model_tq_name == 'only_read':
        noise_model_tq = NoiseModelTQReadoutOnly(
            noise_model_name=configs.qiskit.noise_model_name,
            n_epochs=configs.run.n_epochs,
            prob_schedule=getattr(configs.trainer, 'noise_prob_schedule',
                                  None),
            prob_schedule_separator=getattr(
                configs.trainer, 'noise_prob_schedule_separator', None)
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
        )
    else:
        raise NotImplementedError(configs.trainer.noise_model_tq_name)

    return noise_model_tq
