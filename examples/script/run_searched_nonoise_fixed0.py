import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    # parser.add_argument('--space', type=str)
    args = parser.parse_args()

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{args.dataset}/{args.name}/eval/x2/real/opt2/300.yml',
            '--jobs=2',
            '--run-dir']

    with open(f'logs/nonoise/{args.dataset}.{args.name}.nonoise.fixed0.txt',
              'a') as \
            wfid:
        space = 'barren_s0'
        for mode in ['ldiff_blkexpand.blk8s1.1.1_diff7_chu10_sta40',
                     'ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40'
                     ]:

            exp = f'runs/{args.dataset}.{args.name}.train.searched.scratch' \
                  f'.nonoise.setting0.{space}.{mode}'

            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)
