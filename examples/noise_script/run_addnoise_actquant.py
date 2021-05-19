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
            f'{args.device}/real/opt2/noancilla/act_quant/300_loadop.yml',
            '--jobs=5',
            '--verbose',
            '--run-dir']

    with open(f'logs/{args.device}/{args.dataset}.'
              f'{args.name}.addnoise.actquant.u3cu3_0.multinode.'
              f'{args.noise}.n2b1.n2b2'
              f'.txt',
              'a') as \
            wfid:
        for node_info in ['n2b1',
                          'n2b2',
                          # 'n2b3',
                          # 'n2b4',
                          # 'n3b1',
                          # 'n3b2',
                          # 'n4b1',
                          # 'n4b2'
                          ]:
            for actq in [4, 6, 8, 10]:
                exp = f'runs/{args.dataset}.{args.name}.train.addnoise.' \
                    f'{args.device}.' \
                    f'multinode.u3cu3_0.{node_info}.noi{args.noise}.actq' \
                    f'{actq}.default'

                logger.info(f"running command {pres + [exp]}")
                subprocess.call(pres + [exp], stderr=wfid)
