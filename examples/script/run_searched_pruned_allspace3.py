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

    dataset = 'vowel'
    name = 'four0516'

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{dataset}/{name}/eval/x2/real/opt2/pruned/300.yml',
            '--jobs=2',
            '--run-dir']

    ratios_all = [[0.1, 0.2], [0.2, 0.4], [0.2, 0.4], [0.1, 0.2], [0.1, 0.4]]
    modes = ['plain.blk8s1.1.1',
             'ldiff_blkexpand.blk8s1.1.1_diff7_chu10_sta40',
             'ldiff_blkexpand.blk8s1.1.1_diff7_chu3_sta40',
             'ldiff.blk8s1.1.1_diff7',
             'ldiff_blkexpand.blk4s1.1.1_diff7_chu3_sta40',
             ]

    with open(f"logs/x2/pruned/{dataset}.{name}."
              f"all_space.pruned.txt",
              'a') as wfid:
        for k, space in enumerate([
                      f'u3cu3_s0',
                      f'seth_s0',
                      f'barren_s0',
                      f'farhi_s0',
                      f'maxwell_s0']):
            ratios = ratios_all[k]
            mode = modes[k]
            for ratio in ratios:
                exp = f'runs/{dataset}.{name}.prune.searched.x2.noise.' \
                      f'opt2.setting0.pr{ratio}.{space}.{mode}'
                logger.info(f"running command {pres + [exp]}")
                subprocess.call(pres + [exp], stderr=wfid)
