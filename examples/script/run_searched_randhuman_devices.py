import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str)
    # parser.add_argument('--name', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--nparams', type=int)

    args = parser.parse_args()
    space = 'seth_s0'
    dataset = 'fashion'
    name = 'two36'

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{dataset}/{name}/eval/{args.device}/real/opt2/300.yml',
            '--jobs=4',
            '--run-dir']

    with open(f"logs/{args.device}/randhuman.seth_s0.{dataset}.{name}."
              f"{args.nparams}.txt",
              'a') as \
            wfid:
        exp = f'runs/{dataset}.{name}.train.baseline.' \
              f'{space}.rand.param{args.nparams}'

        logger.info(f"running command {pres + [exp]}")
        subprocess.call(pres + [exp], stderr=wfid)

        exp = f'runs/{dataset}.{name}.train.baseline.' \
              f'{space}.human.param{args.nparams}'

        logger.info(f"running command {pres + [exp]}")
        subprocess.call(pres + [exp], stderr=wfid)


