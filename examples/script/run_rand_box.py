import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--space', type=str)
    args = parser.parse_args()

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{args.dataset}/{args.name}/eval/x2/real/opt2/300.yml',
            '--jobs=5',
            '--run-dir']

    with open(f'logs/x2/box/{args.dataset}.{args.name}.{args.space}.txt',
              'w') as \
            wfid:
        for n_params in [45, 90, 135, 180]:
            for seed in range(8):
                exp = f'runs/{args.dataset}.{args.name}.train.baseline.' \
                      f'{args.space}.rand.param{n_params}.seed{seed}'

                logger.info(f"running command {pres + [exp]}")
                subprocess.call(pres + [exp], stderr=wfid)
