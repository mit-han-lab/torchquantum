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
            f'tq/300_s18400.yml',
            '--jobs=5',
            '--verbose',
            '--run-dir']

    with open(f'logs/tq/{args.dataset}.'
              f'{args.name}.nonoise.u3cu3_0'
              f'.txt',
              'a') as \
            wfid:
        for node in ['n2b1',
                     'n2b2',
                     'n2b3',
                     'n2b4',
                     'n3b1',
                     'n3b2',
                     'n4b1',
                     'n4b2'
                     ]:
            exp = f'runs/{args.dataset}.{args.name}.train.noaddnoise.' \
                  f'lnorm.u3cu3_0.{node}.default'
            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)
