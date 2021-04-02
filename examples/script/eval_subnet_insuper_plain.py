import subprocess
from torchpack.utils.logging import logger
import sys

if __name__ == '__main__':
    pres = ['python',
            'examples/eval.py',
            'examples/configs/mnist/four0123/eval/tq/all.yml',
            '--run-dir=runs/mnist.four0123.train.super'
            '.super4digit_arbitrary_fc1.plain.blk8s2',
            '--ckpt.name',
            'checkpoints/step-18400.pt',
            '--dataset.split=valid']
    with open('eval_subnet_tq_insuper_plain.txt', 'w') as wfid:
        for blk in range(2, 9):
            for ratio in ['0', '0.25', '0.5', '0.75', '1']:
                exp = f"--model.arch.sample_arch=blk{blk}_ratio{ratio}"
                logger.info(f"running command {pres + [exp]}")

                subprocess.run(pres + [exp], stderr=wfid)
