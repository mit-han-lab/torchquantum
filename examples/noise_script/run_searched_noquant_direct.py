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
    parser.add_argument('--nm', type=str)

    args = parser.parse_args()

    valid = '_valid' if args.valid else ''

    path = 'qiskitreadnoi'

    node_dict = {
        'santiago': 'n2b6',
        'x2': 'n2b1',
        'belem': 'n2b3',
        # 'lima': 'n2b2',
        'quito': 'n3b5',
        'athens': 'n3b5',
    }

    last_step_dict = {
        'mnist': {
            'four0123': 18400,
            'two36': 9000,
            'ten': 44600,
        },
        'fashion': {
            'four0123': 18000,
            'two36': 9000,
            'ten': 44600,
        },
        'vowel': {
            'four0516': 10400,
        },
        'cifar': {
            'two68': 7600,
        }
    }

    best_factor = {
        'mnist': {
            'four0123': {
                'santiago': 1.5,
                'x2': 1.5,
                'belem': 0.5
            },
            'two36': {
                'santiago': 1,
                'x2': 0.5,
                'belem': 0.5
            }
        },
        'fashion': {
            'four0123': {
                'santiago': 0.5,
                'x2': 1,
                'belem': 1.5
            },
            # 'two36': {
            #     'santiago': 1,
            #     'x2': 0.5,
            #     'belem': 0.5
            # }
        },
        'vowel': {
            'four0516': {
                'santiago': 0.5,
                'x2': 0.1,
                'belem': 0.5
            },
        }
    }

    best_qlevel = {
        'mnist': {
            'four0123': {
                'santiago': 4,
                'x2': 5,
                'belem': 5
            },
            'two36': {
                'santiago': 4,
                'x2': 5,
                'belem': 5
            }
        },
        'fashion': {
            'four0123': {
                'santiago': 6,
                'x2': 5,
                'belem': 6
            },
            # 'two36': {
            #     'santiago': 1,
            #     'x2': 0.5,
            #     'belem': 0.5
            # }
        },
        'vowel': {
            'four0516': {
                'santiago': 6,
                'x2': 5,
                'belem': 3
            },
        }
    }

    datasets = [
        'mnist',
        'fashion',
        'vowel',
        'mnist',
        # 'fashion',
        # 'cifar',
    ]

    names = [
        'four0123',
        'four0123',
        'four0516',
        'two36',
        # 'two36',
        # 'two68'
    ]



    with open(f'logs/{args.device}/{args.dataset}.'
              f'{args.name}.bnormnolast{valid}.'
              f'{args.nm}.searched_noquant_direct.u3cu3_0'
              f'.{node_dict[args.device]}'
              f'.txt',
              'a') as wfid:
        for dataset, name in zip(datasets, names):
            pres = ['python',
                    'examples/eval.py',
                    f'examples/configs/'
                    f'{dataset}/{name}/eval/'
                    f'{args.device}/real/opt2/noancilla/'
                    f'300_loadop_s{last_step_dict[dataset][name]}'
                    f'{valid}.yml',
                    '--jobs=5',
                    '--verbose',
                    f'--hub={args.hub}',
                    '--run-dir']
            exp = f'runs/{dataset}.{name}.train.addnoise.' \
                  f'bnormnolast.{path}.nothermal.{args.device}.u3cu3_0' \
                  f'.{node_dict[args.device]}.fac' \
                  f'{best_factor[dataset][name][args.device]}' \
                  f'.quant.l{best_qlevel[dataset][name][args.device]}.default'
            logger.info(f"running command {pres + [exp]}")
            if not args.print:
                subprocess.call(pres + [exp], stderr=wfid)
