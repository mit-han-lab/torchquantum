import subprocess
from torchpack.utils.logging import logger
import sys

if __name__ == '__main__':
    pres = ['python',
            'examples/eval.py',
            'examples/configs/mnist/four0123/eval/tq/all.yml',
            '--run-dir=runs/mnist.four0123.train.super'
            '.super4digit_arbitrary_fc1.ldiff_expand'
            '.blk8s2_diff3_chu10_sta40',
            '--ckpt.name',
            'checkpoints/step-18400.pt',
            '--gpu=5',
            '--dataset.split=valid']
    with open('logs/eval_subnet_tq_rand_insuper_expand'
              '.blk8s2_diff3_chu10_sta40.txt', 'w') as wfid:
        for blk in range(1, 9):
            for rand in range(4):
                exp = f"--model.arch.sample_arch=super4digit_arbitrary_fc1_blk{blk}_rand{rand}"
                logger.info(f"running command {pres + [exp]}")

                subprocess.run(pres + [exp], stderr=wfid)
