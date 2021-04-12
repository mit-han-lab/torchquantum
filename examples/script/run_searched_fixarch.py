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

    datasets = ['mnist', 'mnist', 'vowel', 'fashion', 'fashion']
    names = ['four0123', 'two36', 'four0516', 'four0123', 'two36']
    spaces = ['u3cu3_s0', 'farhi_s0', 'maxwell_s0', 'barren_s0', 'seth_s0']

    modes = ['ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40',
             'ldiff_blkexpand.blk8s1.1.1_diff7_chu10_sta40',
             'ldiff_blkexpand.blk4s1.1.1_diff7_chu3_sta40',
             'plain.blk8s1.1.1',
             'plain.blk8s1.1.1',
             ]

    with open(f"logs/x2/ablation/fix_arch.all_space.newbatch.txt", 'a') as \
            wfid:
        for dataset, name, space, mode in zip(datasets, names, spaces, modes):
            pres = ['python',
                    'examples/eval.py',
                    f'examples/configs/'
                    f'{dataset}/{name}/eval/x2/real/opt2/300.yml',
                    '--jobs=4',
                    '--run-dir']

            exp = f'runs/{dataset}.{name}.train.searched.scratch.x2.noise.' \
                  f'opt2.fix_arch.{space}.{mode}'
            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)

