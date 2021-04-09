import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--space', type=str)
    parser.add_argument('--pr', type=float, nargs='+')
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{args.dataset}/{args.name}/eval/x2/real/opt2/pruned/300.yml',
            '--jobs=5',
            '--run-dir']

    with open(f"logs/x2/pruned/{args.dataset}.{args.name}."
              f"{'-'.join(list(map(str, args.pr)))}.{args.space}."
              f"{args.mode}.pruned.txt",
              'w') as wfid:
        for prune_ratio in args.pr:
            if 'maxwell' in args.space:
                n_blk = 4
            else:
                n_blk = 8

            exp = f'runs/{args.dataset}.{args.name}.prune.searched.x2.noise.' \
                  f'opt2.setting0.pr{prune_ratio}.{args.space}.{args.mode}'

            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)
