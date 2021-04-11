import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str)
    # parser.add_argument('--name', type=str)
    # parser.add_argument('--space', type=str)
    # parser.add_argument('--pr', type=float, nargs='+')
    # parser.add_argument('--mode', type=str)
    # args = parser.parse_args()

    datasets = ['fashion']
    names = ['two36']
    spaces = ['seth_s0']

    modes = [
        'plain.blk8s1.1.1',
    ]

    with open(f"logs/x2/ablation/fix_arch_layout2.seth.txt", 'a') as wfid:
        for dataset, name, space, mode in zip(datasets, names, spaces, modes):
            pres = ['python',
                    'examples/eval.py',
                    f'examples/configs/'
                    f'{dataset}/{name}/eval/x2/real/opt2/300.yml',
                    '--jobs=4',
                    '--run-dir']
            for arch in ['arch', 'layout']:
                exp = f'runs/{dataset}.{name}.train.searched.scratch.x2.noise.' \
                      f'opt2.fix_{arch}.{space}.{mode}'
                logger.info(f"running command {pres + [exp]}")
                subprocess.call(pres + [exp], stderr=wfid)

