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

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{args.dataset}/{args.name}/eval/'
            f'{args.device}/real/opt2/noancilla/300_loadop_s18400{valid}.yml',
            '--jobs=5',
            '--verbose',
            '--gpu=2',
            f'--hub={args.hub}',
            '--run-dir']
    if args.nm == 'act':
        path = 'actreadnoi.afternorm'
    elif args.nm == 'phase':
        path = 'phasereadnoi'
    elif args.nm == 'qiskit':
        path = 'qiskitreadnoi'

    with open(f'logs/{args.device}/{args.dataset}.'
              f'{args.name}.bnormnolast{valid}.{args.nm}.nmodelabla.u3cu3_0'
              f'.fac3_100.txt',
              'a') as \
            wfid:
        for node in [
                     # 'n2b1',
                     'n2b2',
                     # 'n2b3',
                     # 'n2b4',
                     # 'n3b1',
                     # 'n3b2',
                     # 'n4b1',
                     # 'n4b2'
                     # 'n3b4',
                     ]:
            for factor in [3, 5, 10, 20, 50, 100]:
                exp = f'runs/{args.dataset}.{args.name}.train.addnoise.' \
                      f'bnormnolast.{path}.{args.device}.u3cu3_0' \
                      f'.{node}.fac{factor}.noquant.default'
                logger.info(f"running command {pres + [exp]}")
                if not args.print:
                    subprocess.call(pres + [exp], stderr=wfid)
