
import sys
import pdb
import tqdm
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
import torchquantum as tq

from torchquantum.datasets import MNIST
from examples.gradient_pruning.q_models import *

# from torchpack import distributed as dist
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from torchpack.utils import io


def log_acc(output_all, target_all, k=1):
    _, indices = output_all.topk(k, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Loss: {loss}")

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
    sampler = torch.utils.data.SequentialSampler(dataset['valid'])
    dataflow = torch.utils.data.DataLoader(
        dataset['valid'],
        sampler=sampler,
        batch_size=configs.run.bsz,
        num_workers=configs.run.workers_per_gpu,
        pin_memory=True)
    
    state_dict = io.load(
        'runs/save_load_example/checkpoints/max-acc-valid.pt',
        map_location='cpu')
    
    model = QMultiFCModel0(configs.model.arch)
    model.load_state_dict(state_dict['model'], strict=False)

    if configs.qiskit.use_qiskit_train or configs.qiskit.use_qiskit_valid:
        from torchquantum.plugins import QiskitProcessor
        processor = QiskitProcessor(use_real_qc=configs.qiskit.use_real_qc, n_shots=configs.qiskit.n_shots, backend_name=configs.qiskit.backend_name)
        model.set_qiskit_processor(processor)

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Size: {total_params}')
    
    with torch.no_grad():
        target_all = None
        output_all = None
        for feed_dict in tqdm.tqdm(dataflow):
            if configs.run.device == 'gpu':
                # pdb.set_trace()
                inputs = feed_dict[configs.dataset.input_name].cuda(non_blocking=True)
                targets = feed_dict[configs.dataset.target_name].cuda(non_blocking=True)
            else:
                inputs = feed_dict[configs.dataset.input_name]
                targets = feed_dict[configs.dataset.target_name]

            outputs = model(inputs, use_qiskit=configs.qiskit.use_qiskit_valid)

            if target_all is None:
                target_all = targets
                output_all = outputs
            else:
                target_all = torch.cat([target_all, targets], dim=0)
                output_all = torch.cat([output_all, outputs], dim=0)

        log_acc(output_all, target_all)

    
    

if __name__ == '__main__':
    main()