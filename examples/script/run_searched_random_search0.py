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

    datasets = ['mnist',]
    names = ['four0123',]
    spaces = ['u3cu3_s0',]

    modes = ['ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40',
             ]

    with open(f"logs/x2/ablation/random_search.u3cu3_s0.txt",
              'a') as \
            wfid:
        for dataset, name, space, mode in zip(datasets, names, spaces, modes):
            pres = ['python',
                    'examples/eval.py',
                    f'examples/configs/'
                    f'{dataset}/{name}/eval/x2/real/opt2/300.yml',
                    '--jobs=4',
                    '--run-dir']

            exp = f'runs/{dataset}.{name}.train.searched.scratch.x2.noise.' \
                  f'opt2.random_search.{space}.{mode}'
            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)

