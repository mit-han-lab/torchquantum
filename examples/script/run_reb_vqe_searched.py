import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--space', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--fix', action='store_true')

    args = parser.parse_args()

    setting = 'fix_layout' if args.fix else 'setting0'

    n_blk = 8

    devices = [
               # 'rome',
               # 'lima',
        # 'quito',
        # 'x2',
        # 'belem',
        # 'santiago',
        'athens'
    ]
    mode = f'plain.blk{n_blk}s1.1.1'

    with open(f'logs/reb/topoerror.{setting}.{args.dataset}.{args.name}'
              f'.{args.space}.athens.txt',
              'a') as \
            wfid:
        for device in devices:

            pres = ['python',
                    'examples/eval.py',
                    f'examples/configs/'
                    f'{args.dataset}/{args.name}/eval/'
                    f'{device}/real/opt2/all.yml',
                    '--jobs=1',
                    '--run-dir']

            exp = f'runs/{args.dataset}.{args.name}.train.searched.scratch' \
                  f'.{device}.noise.opt2.{setting}.{args.space}.{mode}'

            logger.info(f"running command {pres + [exp]}")

            if not args.print:
                subprocess.call(pres + [exp], stderr=wfid)
