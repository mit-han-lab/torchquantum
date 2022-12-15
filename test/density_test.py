import torchquantum as tq
import pdb
import numpy as np
import torch



if __name__ == '__main__':
    density = tq.DensityMatrix(
        n_wires=4,
        bsz=1
    )
    density.print_2d(0)