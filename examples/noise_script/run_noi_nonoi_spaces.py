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


    spaces = [
        'seth_0',
        'barren_0',
        'farhi_0',
        'maxwell_0'
    ]

    # node_dict = {
    #     'santiago': 'n2b6',
    #     'x2': 'n2b1',
    #     'belem': 'n2b3',
    #     'lima': 'n2b2',
        # 'quito': 'n3b5',
        # 'athens': 'n3b5',
    # }

    # nodes = [
    #     'n2b2',
    #     'n2b4',
    # ]
    node_dict = {
        'santiago': 'n2b2',
        'x2': 'n2b1'
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



    with open(f'logs/{args.device}/{args.dataset}.'
              f'{args.name}.bnormnolast{valid}.'
              f'{args.nm}.noi_nonoi_spaces'
              f'.fac1_ql5.nothermal.txt',
              'a') as wfid:
        for space in spaces:
            # for node in nodes:
                # if factor == 1 and level < 5:
                #     continue
            pres = ['python',
                    'examples/eval.py',
                    f'examples/configs/'
                    f'{args.dataset}/{args.name}/eval/'
                    f'{args.device}/real/opt2/noancilla/'
                    f'300_s{last_step_dict[args.dataset][args.name]}'
                    f'{valid}.yml',
                    '--jobs=5',
                    '--verbose',
                    f'--hub={args.hub}',
                    '--run-dir']

            exp = f'runs/{args.dataset}.{args.name}.train.noaddnoise.' \
                  f'nonorm.{space}' \
                  f'.{node_dict[args.device]}' \
                  f'.default'
            logger.info(f"running command {pres + [exp]}")
            if not args.print:
                subprocess.call(pres + [exp], stderr=wfid)

            pres = ['python',
                    'examples/eval.py',
                    f'examples/configs/'
                    f'{args.dataset}/{args.name}/eval/'
                    f'{args.device}/real/opt2/noancilla/act_quant/'
                    f'300_loadop_s{last_step_dict[args.dataset][args.name]}'
                    f'{valid}.yml',
                    '--jobs=5',
                    '--verbose',
                    f'--hub={args.hub}',
                    '--run-dir']

            exp = f'runs/{args.dataset}.{args.name}.train.addnoise.' \
                  f'bnormnolast.{path}.nothermal.{args.device}.{space}' \
                  f'.{node_dict[args.device]}.fac1.quant' \
                  f'.l5.default'
            logger.info(f"running command {pres + [exp]}")
            if not args.print:
                subprocess.call(pres + [exp], stderr=wfid)


