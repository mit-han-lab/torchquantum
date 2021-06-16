import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--space', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--hub', type=str, default=None)

    parser.add_argument('--print', action='store_true')
    args = parser.parse_args()

    if 'maxwell' in args.space:
        n_blk = 4
        n_rand = 16
    else:
        n_blk = 8
        n_rand = 5

    mode = args.mode
    # mode = f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu3_sta40'
    # mode = f'plain.blk{n_blk}s1.1.1'
    # f'ldiff.blk{n_blk}s1.1.1_diff7',
    # f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu3_sta40',
    # f'ldiff_blkexpand.blk{n_blk}s1.1.1_diff7_chu10_sta40'

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/{args.dataset}/{args.name}/eval/tq/all.yml',
            f'--run-dir=runs/{args.dataset}.{args.name}.train.super'
            f'.{args.space}.{mode}',
            '--ckpt.name',
            'checkpoints/step-18400.pt',
            '--dataset.split=valid',
            f'--hub={args.hub}',
            f'--gpu={args.gpu}']
    with open(f'logs/reb/eval_subnet_tq_insuper_{args.dataset}.{args.name}.{args.space}.{mode}.txt', 'w') as \
            wfid:
        for blk in range(1, n_blk + 1):
            for ratio in ['0', '0.3', '0.6', '1']:
                exp = f"--model.arch.sample_arch=blk{blk}_ratio{ratio}"
                logger.info(f"running command {pres + [exp]}")

                if not args.print:
                    subprocess.run(pres + [exp], stderr=wfid)

        for blk in range(1, n_blk + 1):
            for rand in range(n_rand):
                exp = f"--model.arch.sample_arch=sharefront0_blk{blk}_rand{rand}"
                logger.info(f"running command {pres + [exp]}")

                if not args.print:
                    subprocess.run(pres + [exp], stderr=wfid)


