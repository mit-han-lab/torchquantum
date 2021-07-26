import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    # parser.add_argument('--space', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--fix', action='store_true')
    parser.add_argument('--hub', type=str, default=None)

    args = parser.parse_args()

    setting = 'fix_layout' if args.fix else 'setting0'

    noise_setting = 'estnoise' if args.device in ['manhattan',
                                                  'montreal',
                                                  'melbourne'] else 'noise'

    # mode = f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu3_sta40'

    with open(f'logs/reb/vqe.op200.uccsd.step1000.{setting}.{args.dataset}.{args.name}.'
              f'u3cu3_s0.{args.device}.txt',
              'a') as \
            wfid:
        # for mode in [f'plain.blk{n_blk}s1.1.1',
        #              f'ldiff.blk{n_blk}s1.1.1_diff7',
        #              f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu3_sta40',
        #              f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu10_sta40',]:
        for k, space in enumerate([
            f'u3cu3_s0',
            # f'seth_s0',
            # f'barren_s0',
            # f'farhi_s0',
            # f'maxwell_s0'
        ]):
            if 'maxwell' in space:
                n_blk = 4
            else:
                n_blk = 8

            mode = f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu10_sta40'

            pres = ['python',
                    'examples/eval.py',
                    f'examples/configs/'
                    f'{args.dataset}/step1000/{args.name}/eval/'
                    f'{args.device}/real/opt2/all.yml',
                    '--jobs=5',
                    f'--hub={args.hub}',
                    '--gpu=7',
                    '--run-dir']

            exp = f'runs/{args.dataset}.step1000.{args.name}.train.baseline' \
                  f'.uccsd.op200/'

            logger.info(f"running command {pres + [exp]}")
            if not args.print:
                subprocess.call(pres + [exp], stderr=wfid)
            else:
                subprocess.call(['ls'] + [exp+'checkpoints'])

