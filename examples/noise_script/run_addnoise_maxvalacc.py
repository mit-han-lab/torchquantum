import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)

    parser.add_argument('--device', type=str)
    parser.add_argument('--blk', type=int, nargs='+')

    args = parser.parse_args()

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{args.dataset}/{args.name}/eval/'
            f'{args.device}/real/opt2/noancilla/300_load_op_list.yml',
            '--jobs=5',
            '--verbose',
            '--run-dir']

    with open(f'logs/{args.device}/{args.dataset}.'
              f"{args.name}.addnoise.maxvalacc.u3cu3_0.blk."
              f"{'-'.join(list(map(str, args.blk)))}"
              f'.txt',
              'a') as \
            wfid:
        for blk in args.blk:
            for noise in ['_orig', '0.005', '0.01', '0.02', '0.05', '0.1',
                          '0.2', '0.5']:
                exp = f'runs/{args.dataset}.{args.name}.train.addnoise.' \
                      f'{args.device}.u3cu3_0.blk{blk}.addnoise{noise}' \

                logger.info(f"running command {pres + [exp]}")
                subprocess.call(pres + [exp], stderr=wfid)
