import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)

    parser.add_argument('--device', type=str)
    parser.add_argument('--hub', type=str, default=None)
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--node', type=str, default=None)

    args = parser.parse_args()

    valid = '_valid' if args.valid else ''

    node_dict = {
        'santiago': 'n2b3',
        'x2': 'n2b3',
        # 'belem': 'n2b3',
        'lima': 'n2b3',
        # 'quito': 'n3b3',
        # 'athens': 'n3b6',
    }

    last_step_dict = {
        'mnist': {
            'four0123': 18400,
            'two36': 9000,
        },
        'fashion': {
            'four0123': 18000,
            'two36': 9000,
        },
        'vowel': {
            'four0516': 10400,
        },
        'cifar': {
            'two68': 7600,
        }
    }

    appen = '' if args.node is None else f".{args.node}"

    with open(f'logs/multi/{args.dataset}.'
              f'{args.name}.nonoise_baselines{valid}.u3cu3_0{appen}'
              f'.motivation.bnormnolast.txt',
              'a') as \
            wfid:
        for setting in [
            # 'nonorm',
            'bnormnolast'
        ]:
            for device in [
                'x2',
                'lima',
                'santiago',
                # 'belem',
                # 'quito'
            ]:

                last_step = last_step_dict[args.dataset][args.name]
                pres = ['python',
                        'examples/eval.py',
                        f'examples/configs/'
                        f'{args.dataset}/{args.name}/eval/'
                        f'{device}/real/opt2/noancilla/'
                        f'300_s{last_step}{valid}.yml',
                        '--jobs=5',
                        '--verbose',
                        '--gpu=2',
                        f'--hub={args.hub}',
                        '--run-dir']

                exp = f'runs/{args.dataset}.{args.name}.train.noaddnoise.' \
                      f'{setting}.u3cu3_0.{node_dict[device]}.default'
                logger.info(f"running command {pres + [exp]}")

                if not args.print:
                    subprocess.call(pres + [exp], stderr=wfid)
