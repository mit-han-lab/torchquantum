import argparse
import os
import pdb
import torch
import torch.backends.cudnn
import tqdm
import torch.nn.functional as F

from torchpack.utils import io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from core import builder
from torchquantum.utils import legalize_unitary
from qiskit import IBMQ


def main() -> None:
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run_dir', metavar='DIR', help='run directory')
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    args, opts = parser.parse_known_args()

    configs.load(os.path.join(args.run_dir, 'metainfo', 'configs.yaml'))
    configs.load(args.config, recursive=True)
    configs.update(opts)

    if configs.debug.pdb or args.pdb:
        pdb.set_trace()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if configs.run.device == 'gpu':
        device = torch.device('cuda')
    elif configs.run.device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError(configs.run.device)

    logger.info(f'Evaluation started: "{args.run_dir}".' + '\n' + f'{configs}')

    if configs.qiskit.use_qiskit:
        IBMQ.load_account()
        if configs.run.bsz == 'qiskit_max':
            configs.run.bsz = IBMQ.get_provider(hub='ibm-q').get_backend(
                configs.qiskit.backend_name).configuration().max_experiments

    dataset = builder.make_dataset()
    sampler = torch.utils.data.SequentialSampler(dataset['test'])
    dataflow = torch.utils.data.DataLoader(
        dataset['test'],
        sampler=sampler,
        batch_size=configs.run.bsz,
        num_workers=configs.run.workers_per_gpu,
        pin_memory=True)

    state_dict = io.load(
        os.path.join(args.run_dir, 'checkpoints', 'max-acc-valid.pt'),
        map_location=device)
    model = state_dict['model_arch']

    if configs.legalize_unitary:
        legalize_unitary(model)
    model.to(device)
    model.eval()
    model.load_state_dict(state_dict['model'])

    if configs.qiskit.use_qiskit:
        qiskit_processor = builder.make_qiskit_processor()
        model.set_qiskit_processor(qiskit_processor)

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
                outputs = model.forward_qiskit(inputs, targets)
            else:
                outputs = model(inputs)

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
    logger.info(f"Accuracy: {corrects / size}")
    logger.info(f"Loss: {F.nll_loss(output_all, target_all)}")


if __name__ == '__main__':
    main()
