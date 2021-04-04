import subprocess
from torchpack.utils.logging import logger


if __name__ == '__main__':
    pres = ['python',
            'examples/eval.py',
            'examples/configs/mnist/four0123/eval/lima/real/opt2/300.yml',
            '--jobs=1',
            '--run-dir']
    with open('logs/eval_subnet_lima_real_300_opt2_blk5.8ratio1.txt', 'w') as \
            wfid:
        for blk in [5, 8]:
            for ratio in ['1']:
                exp = f'runs/mnist.four0123.train.baseline' \
                      f'.super4digit_arbitrary_fc1.blk8s1.blk{blk}_ratio' \
                      f'{ratio}/'
                logger.info(f"running command {pres + [exp]}")

                subprocess.call(pres + [exp], stderr=wfid)
