import subprocess
from torchpack.utils.logging import logger


if __name__ == '__main__':
    pres = ['python',
            'examples/eval.py',
            'examples/configs/mnist/four0123/eval/x2/noise/opt2/all.yml',
            '--get_n_params=True',
            '--run-dir']
    with open('logs/eval_subnet_rand_nparams.txt', 'w') as wfid:
        for blk in range(1, 9):
            for rand in range(4):
                exp = f'runs/mnist.four0123.train.baseline' \
                      f'.super4digit_arbitrary_fc1.blk8s1.blk{blk}_rand' \
                      f'{rand}/'
                logger.info(f"running command {pres + [exp]}")

                subprocess.call(pres + [exp], stderr=wfid)
