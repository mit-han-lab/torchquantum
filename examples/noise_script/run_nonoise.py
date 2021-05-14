import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)

    parser.add_argument('--device', type=str)
    args = parser.parse_args()

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{args.dataset}/{args.name}/eval/'
            f'{args.device}/real/opt2/noancilla/300.yml',
            '--jobs=5',
            '--run-dir']

    with open(f'logs/{args.device}/{args.dataset}.{args.name}'
              f'.txt',
              'a') as \
            wfid:
        for n_blk in range(4, 9):
            exp = f'runs/{args.dataset}.{args.name}.train.noaddnoise' \
                  f'.u3cu3_0.blk{n_blk}.default'

            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)
