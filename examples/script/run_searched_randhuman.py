import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    # parser.add_argument('--space', type=str)
    parser.add_argument('--nparams', type=int, nargs='+')
    args = parser.parse_args()

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{args.dataset}/{args.name}/eval/x2/real/opt2/300.yml',
            '--jobs=4',
            '--run-dir']

    # params = [7, 26, 15, 17]

    with open(f"logs/x2/randhuman/{args.dataset}.{args.name}."
              f"{'-'.join(list(map(str, args.nparams)))}.randhuman.txt",
              'a') as \
            wfid:
        for k, space in enumerate([f'u3cu3_s0',
                                   f'seth_s0',
                                   f'barren_s0',
                                   f'farhi_s0',
                                   f'maxwell_s0']):

            if 'maxwell' in space:
                n_blk = 4
            else:
                n_blk = 8

            exp = f'runs/{args.dataset}.{args.name}.train.baseline.' \
                  f'{space}.rand.param{args.nparams[k]}'

            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)

        for k, space in enumerate([f'u3cu3_s0',
                                   f'seth_s0',
                                   f'barren_s0',
                                   f'farhi_s0',
                                   f'maxwell_s0']):

            if 'maxwell' in space:
                n_blk = 4
            else:
                n_blk = 8

            exp = f'runs/{args.dataset}.{args.name}.train.baseline.' \
                  f'{space}.human.param{args.nparams[k]}'

            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)
