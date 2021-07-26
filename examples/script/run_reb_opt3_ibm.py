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
    nparam = {
        'mnist': {
            'four0123': {
                'u3cu3_s0': 36,
                'seth_s0': 7,
                'barren_s0': 26,
                'farhi_s0': 15,
                'maxwell_s0': 17,
                'ibmbasis_a0': 14,
            },
            'two36': {
                'u3cu3_s0': 51,
                'seth_s0': 8,
                'barren_s0': 24,
                'farhi_s0': 10,
                'maxwell_s0': 19,
                'ibmbasis_a0': 10,
            }
        },
        'fashion': {
            'four0123': {
                'u3cu3_s0': 51,
                'seth_s0': 23,
                'barren_s0': 34,
                'farhi_s0': 7,
                'maxwell_s0': 18,
                'ibmbasis_a0': 13,
            },
            'two36': {
                'u3cu3_s0': 36,
                'seth_s0': 14,
                'barren_s0': 14,
                'farhi_s0': 10,
                'maxwell_s0': 21,
                'ibmbasis_a0': 5,
            }
        },
        'vowel': {
            'four0516': {
                'u3cu3_s0': 75,
                'seth_s0': 7,
                'barren_s0': 23,
                'farhi_s0': 17,
                'maxwell_s0': 15,
                'ibmbasis_a0': 20,
            }
        }
    }

    # mode = f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu3_sta40'

    with open(f'logs/reb/ibm.opt3.{setting}.{args.dataset}.{args.name}'
              f'.allspace.{args.device}.txt',
              'a') as \
            wfid:
        # for mode in [f'plain.blk{n_blk}s1.1.1',
        #              f'ldiff.blk{n_blk}s1.1.1_diff7',
        #              f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu3_sta40',
        #              f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu10_sta40',]:
        for k, space in enumerate([
            # f'u3cu3_s0',
            # f'seth_s0',
            # f'barren_s0',
            # f'farhi_s0',
            # f'maxwell_s0'
            'ibmbasis_a0'
        ]):
            pres = ['python',
                    'examples/eval.py',
                    f'examples/configs/'
                    f'{args.dataset}/{args.name}/eval/{args.device}/real/opt3/sabre/300.yml',
                    '--jobs=5',
                    f'--hub={args.hub}',
                    '--run-dir']

            exp = f'runs/{args.dataset}.{args.name}' \
                  f'.train.baseline.{space}.human.param{nparam[args.dataset][args.name][space]}/'

            logger.info(f"running command {pres + [exp]}")
            if not args.print:
                subprocess.call(pres + [exp], stderr=wfid)
            else:
                subprocess.call(['ls'] + [exp+'checkpoints'])

