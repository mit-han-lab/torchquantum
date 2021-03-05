import argparse
import os
import pdb
import torch

from torchquantum.utils import Timer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', action='store_true', help='pdb')
    parser.add_argument('--gpu', type=str, help='gpu ids', default=None)
    args, opts = parser.parse_known_args()

    if args.pdb:
        pdb.set_trace()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    times = 10000
    input1 = torch.randn(2, 2).cuda()
    input2 = torch.randn(256, 2, 2).cuda()

    with Timer(name='matmul', times=times):
        for _ in range(times):
            torch.matmul(input1, input2)

    with Timer(name='bmm', times=times):
        for _ in range(times):
            torch.bmm(input1.expand(input2.shape), input2)

    with Timer(name='einsum', times=times):
        for _ in range(times):
            torch.einsum('ab,zbc->zac', input1, input2)