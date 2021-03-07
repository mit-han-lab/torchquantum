import torch
import torch.nn as nn

from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler

__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer',
    'make_scheduler'
]


def make_dataset() -> Dataset:
    if configs.dataset.name == 'mnist':
        from .datasets import MNIST
        dataset = MNIST(
            root=configs.dataset.root,
            train_valid_split_ratio=configs.dataset.train_valid_split_ratio
        )
    elif configs.dataset.name == 'mnist_small':
        from .datasets import MNIST
        dataset = MNIST(
            root=configs.dataset.root,
            train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
            center_crop=configs.dataset.center_crop,
            resize=configs.dataset.resize
        )
    elif configs.dataset.name == 'layer_regression':
        from .datasets import LayerRegression
        dataset = LayerRegression()
    elif configs.dataset.name == 'mnist_binarized':
        from .datasets import MNISTBinarized
        dataset = MNISTBinarized(
            root=configs.dataset.root,
            train_valid_split_ratio=configs.dataset.train_valid_split_ratio
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
        model = model_dict[configs.model.name]()
    elif configs.model.name == 'layer_regression':
        from .models import LayerRegression
        model = LayerRegression()
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
