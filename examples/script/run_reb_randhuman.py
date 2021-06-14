import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--space', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--nparam', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--fix', action='store_true')

    args = parser.parse_args()

    setting = 'fix_layout' if args.fix else 'setting0'

    # if 'maxwell' in args.space:
    #     n_blk = 4
    # else:
    #     n_blk = 8

    # devices = [
        # 'rome',
               # 'lima',
        # 'quito',
        # 'x2',
        # 'belem',
        # 'santiago',
        # 'athens'
        #        ]
    # mode = f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu3_sta40'


    with open(f'logs/reb/randhuman.{args.dataset}.{args.name}.{args.device}.'
              f'{args.space}.{args.nparam}.{args.mode}.txt',
              'a') as \
            wfid:
        # for mode in [f'plain.blk{n_blk}s1.1.1',
        #              f'ldiff.blk{n_blk}s1.1.1_diff7',
        #              f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu3_sta40',
        #              f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu10_sta40',]:
        exps = [
            f'runs/{args.dataset}.{args.name}.train.searched.scratch.nonoise.setting0.{args.space}.{args.mode}/',
            f'runs/{args.dataset}.{args.name}.train.baseline.{args.space}.rand.param{args.nparam}.seed0/',
            f'runs/{args.dataset}.{args.name}.train.baseline.{args.space}.human.param{args.nparam}/',
        ]
        for exp in exps:
            pres = ['python',
                    'examples/eval.py',
                    f'examples/configs/'
                    f'{args.dataset}/{args.name}/eval/{args.device}/real/opt2/300.yml',
                    '--jobs=5',
                    '--run-dir']

            # exp = f'runs/{args.dataset}.{args.name}.train.searched.scratch' \
            #       f'.{device}.noise.opt2.{setting}.{args.space}.{mode}'

            logger.info(f"running command {pres + [exp]}")
            if not args.print:
                subprocess.call(pres + [exp], stderr=wfid)

