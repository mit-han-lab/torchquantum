import torch
import torch.nn as nn

from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler
from torchpack.callbacks import (InferenceRunner, MeanAbsoluteError, MaxSaver,
                                 Saver, SaverRestore, CategoricalAccuracy)
from .callbacks import LegalInferenceRunner, SubnetInferenceRunner, NLLError
from torchquantum.plugins import QiskitProcessor


__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler', 'make_callbacks', 'make_qiskit_processor'
]


def make_dataset() -> Dataset:
    if configs.dataset.name == 'mnist':
        from .datasets import MNIST
        dataset = MNIST(
            root=configs.dataset.root,
            train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
            center_crop=configs.dataset.center_crop,
            resize=configs.dataset.resize,
            binarize=configs.dataset.binarize,
            binarize_threshold=configs.dataset.binarize_threshold,
            digits_of_interest=configs.dataset.digits_of_interest,
            n_test_samples=configs.dataset.n_test_samples,
        )
    elif configs.dataset.name == 'layer_regression':
        from .datasets import LayerRegression
        dataset = LayerRegression()
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
        model = model_dict[configs.model.name]()
    elif configs.model.name == 'layer_regression':
        from .models import LayerRegression
        model = LayerRegression()
    elif configs.model.name.startswith('super_'):
        from .models.super_models import model_dict
        model = model_dict[configs.model.name]()
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
        else:
            raise NotImplementedError(subcallback['metrics'])
    return subcallbacks


def make_callbacks(dataflow):
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
        elif config['callback'] == 'SaverRestore':
            callback = SaverRestore()
        elif config['callback'] == 'Saver':
            callback = Saver()
        elif config['callback'] == 'MaxSaver':
            callback = MaxSaver(config['name'])
        else:
            raise NotImplementedError(config['callback'])
        callbacks.append(callback)

    return callbacks


def make_qiskit_processor():
    processor = QiskitProcessor(
        use_real_qc=configs.qiskit.use_real_qc,
        backend_name=configs.qiskit.backend_name,
        noise_model_name=configs.qiskit.noise_model_name,
        coupling_map_name=configs.qiskit.coupling_map_name,
        basis_gates_name=configs.qiskit.basis_gates_name,
        n_shots=configs.qiskit.n_shots,
        initial_layout=configs.qiskit.initial_layout,
        seed_transpiler=configs.qiskit.seed_transpiler,
        seed_simulator=configs.qiskit.seed_simulator,
        optimization_level=configs.qiskit.optimization_level
    )
    return processor
