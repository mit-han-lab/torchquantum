import subprocess
from torchpack.utils.logging import logger
import sys

if __name__ == '__main__':
    pres = ['python',
            'examples/eval.py',
            'examples/configs/mnist/four0123/eval/tq/all.yml',
            '--run-dir']
    with open('logs/x2/box/noisefree_tq_rand.txt', 'a') as wfid:
        for n_params in [45, 90, 135, 180]:
            for seed in range(8):
                exp = f'runs/mnist.four0123.train.baseline' \
                      f'.u3cu3_s0.rand.param{n_params}.seed{seed}'
                logger.info(f"running command {pres + [exp]}")

                subprocess.run(pres + [exp], stderr=wfid)
