import sys
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
import torchquantum as tq

from torch.optim.lr_scheduler import CosineAnnealingLR

from torchquantum.datasets import MNIST
from examples.gradient_pruning.q_models import *
from torchpack.callbacks import (InferenceRunner, MeanAbsoluteError,
                                 MaxSaver, MinSaver,
                                 Saver, SaverRestore, CategoricalAccuracy)
from examples.gradient_pruning.callbacks import LegalInferenceRunner, SubnetInferenceRunner, \
    NLLError, AddNoiseInferenceRunner, GradRestore

# from torchpack import distributed as dist
from torchpack.environ import set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

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

    return callbacks

def main():
    configs.load('examples/save_load_example/configs.yml')

    if configs.debug.set_seed:
        torch.manual_seed(configs.debug.seed)
        np.random.seed(configs.debug.seed)

    device = torch.device('cuda')
    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)
    
    dataset = MNIST(
        root='./mnist_data',
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[0, 1, 2, 3],
        n_test_samples=30,
        n_train_samples=50,
        n_valid_samples=30,
    )
    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.run.bsz,
            sampler=sampler,
            num_workers=configs.run.workers_per_gpu,
            pin_memory=True)

    model = QMultiFCModel0(configs.model.arch)

    if configs.qiskit.use_qiskit_train or configs.qiskit.use_qiskit_valid:
        from torchquantum.plugins import QiskitProcessor
        processor = QiskitProcessor(use_real_qc=configs.qiskit.use_real_qc, n_shots=configs.qiskit.n_shots, backend_name=configs.qiskit.backend_name)
        model.set_qiskit_processor(processor)

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Size: {total_params}')

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configs.optimizer.lr,
        weight_decay=configs.optimizer.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=configs.run.n_epochs)

    from examples.gradient_pruning.trainers import ParamsShiftTrainer
    trainer = ParamsShiftTrainer(model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler)

    trainer.set_use_qiskit(configs)
    run_dir = 'runs/save_load_example/'
    set_run_dir(run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))

    logger.info(f'Training started: "{run_dir}".' + '\n' +
        f'{configs}')

    callbacks = make_callbacks(dataflow)

    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.run.n_epochs,
        callbacks=callbacks)
    
    

if __name__ == '__main__':
    main()