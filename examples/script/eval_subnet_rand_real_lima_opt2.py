import subprocess
from torchpack.utils.logging import logger


if __name__ == '__main__':
    pres = ['python',
            'examples/eval.py',
            'examples/configs/mnist/four0123/eval/lima/real/opt2/300.yml',
            '--jobs=5',
            '--run-dir']
    with open('logs/eval_subnet_rand_lima_real_300_opt2_blk5.ran1.2.txt',
              'w') as \
            wfid:
        for blk in range(5, 6):
            for rand in range(1, 3):
                exp = f'runs/mnist.four0123.train.baseline' \
                      f'.super4digit_arbitrary_fc1.blk8s1.blk{blk}_rand' \
                      f'{rand}/'
                logger.info(f"running command {pres + [exp]}")

                subprocess.call(pres + [exp], stderr=wfid)
