import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)

    parser.add_argument('--device', type=str)
    parser.add_argument('--noise', type=str)

    args = parser.parse_args()

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{args.dataset}/{args.name}/eval/'
            f'{args.device}/real/opt2/noancilla/300_loadop_s18400.yml',
            '--jobs=5',
            '--verbose',
            '--run-dir']

    with open(f'logs/{args.device}/{args.dataset}.'
              f'{args.name}.addfromqiskitnoise.'
              f'{args.noise}.indecre.u3cu3_0.n2b2'
              f'.txt',
              'a') as \
            wfid:
        for indecre in ['incre', 'decre', 'indecre'
            # 'orig',
            #           '0.005',
                      # '0.01',
                      # '0.02',
                      # '0.05',
                      # '0.1',
                      # '0.2',
                      # '0.5'
        ]:

            exp = f'runs/{args.dataset}.{args.name}.train.addnoise.' \
                  f'{args.device}.multinode.u3cu3_0.n2b2.noi' \
                  f'{args.noise}.noactq.noi_{indecre}'
            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)
