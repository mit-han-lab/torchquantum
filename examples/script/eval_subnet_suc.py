import subprocess
from torchpack.utils.logging import logger


if __name__ == '__main__':
    pres = ['python',
            'examples/eval.py',
            'examples/configs/mnist/four0123/eval/x2/noise/opt2/all.yml',
            '--qiskit.est_success_rate=True',
            '--run-dir']
    with open('logs/sfsuper/eval_subnet_suc_ratiorand.txt', 'w') as wfid:

        for blk in range(1, 9):
            for rand in range(4):
                exp = f'runs/mnist.four0123.train.baseline' \
                      f'.u3cu3_s0.subnet.blk{blk}_rand' \
                      f'{rand}/'
                logger.info(f"running command {pres + [exp]}")

                subprocess.call(pres + [exp], stderr=wfid)

        for blk in range(1, 9):
            for ratio in ['0', '0.3', '0.6', '1']:
                exp = f'runs/mnist.four0123.train.baseline' \
                      f'.u3cu3_s0.subnet.blk{blk}_ratio' \
                      f'{ratio}/'
                logger.info(f"running command {pres + [exp]}")

                subprocess.call(pres + [exp], stderr=wfid)
