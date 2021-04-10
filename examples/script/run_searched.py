import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--space', type=str)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{args.dataset}/{args.name}/eval/{args.device}/real/opt2/300.yml',
            '--jobs=5',
            '--run-dir']

    if 'maxwell' in args.space:
        n_blk = 4
    else:
        n_blk = 8

    with open(f'logs/{args.device}/{args.dataset}.{args.name}'
              f'.{args.space}.txt',
              'w') as \
            wfid:
        for mode in [f'plain.blk{n_blk}s1.1.1',
                     f'ldiff.blk{n_blk}s1.1.1_diff7',
                     f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu3_sta40',
                     f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu10_sta40',]:
            exp = f'runs/{args.dataset}.{args.name}.train.searched.scratch' \
                  f'.{args.device}.noise.opt2.setting0.{args.space}.{mode}'

            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)
