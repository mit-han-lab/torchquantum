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

    if args.nm == 'act':
        path = 'actreadnoi.afternorm'
    elif args.nm == 'phase':
        path = 'phasereadnoi'
    elif args.nm == 'qiskit':
        path = 'qiskitreadnoi'

    node_dict = {
        'santiago': 'n2b6',
        'x2': 'n2b1',
        'belem': 'n2b3',
        # 'lima': 'n2b2',
        'quito': 'n3b5',
        # 'athens': 'n3b6',
    }

    last_step_dict = {
        'mnist': {
            'four0123': 18400,
        },
        'fashion': {
            'four0123': 18000,
        }
    }

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

    with open(f'logs/{args.device}/{args.dataset}.'
              f'{args.name}.bnormnolast{valid}.'
              f'{args.nm}.noi_qlevel_abla.u3cu3_0.{node_dict[args.device]}'
              f'.fac0.05-0.1-0.2-2_ql3-4-5-6.nothermal.txt',
              'a') as wfid:
        for factor in [0.05, 0.1, 0.2, 2]:
            for level in [3, 4, 5, 6]:
                exp = f'runs/{args.dataset}.{args.name}.train.addnoise.' \
                      f'bnormnolast.{path}.nothermal.{args.device}.u3cu3_0' \
                      f'.{node_dict[args.device]}.fac{factor}.quant' \
                      f'.l{level}.default'
                logger.info(f"running command {pres + [exp]}")
                if not args.print:
                    subprocess.call(pres + [exp], stderr=wfid)
