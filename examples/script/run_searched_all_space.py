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

    with open(f'logs/x2/{args.dataset}.{args.name}.all_space.txt', 'w') as \
            wfid:
        for space in [f'u3cu3_s0',
                      f'seth_s0',
                      f'barren_s0',
                      f'farhi_s0',
                      f'maxwell_s0']:
            if 'maxwell' in space:
                n_blk = 4
            else:
                n_blk = 8
            for mode in [f'plain.blk{n_blk}s1.1.1',
                         f'ldiff.blk{n_blk}s1.1.1_diff7',
                         f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu3_sta40',
                         f'ldiff_blkexpand'
                         f'.blk{n_blk}s1.1.1_diff7_chu10_sta40',]:
                exp = f'runs/{args.dataset}' \
                     f'.{args.name}.train.searched.scratch' \
                      f'.x2.noise.opt2.setting0.{space}.{mode}'

                logger.info(f"running command {pres + [exp]}")
                subprocess.call(pres + [exp], stderr=wfid)
