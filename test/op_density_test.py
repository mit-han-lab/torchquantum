import unittest
from torchquantum.macro import C_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import torch


def all_zero_state(n_wires):
    _matrix = torch.zeros(2 ** (2*n_wires), dtype=C_DTYPE)
    _matrix[0] = 1 + 0j
    _matrix = torch.reshape(_matrix, [1]+[2]*(2*n_wires))
    return _matrix



class Test(unittest.TestCase):
    def setUp(self):
        print('Density Matrix test start')

    def tearDown(self):
        print('Density Matrix test end')

    def test_special_case(self):
        state = tq.QuantumState(
            n_wires=4,
            bsz=1
        )
        state.hadamard(wires=2, inverse=False)
        state.hadamard(wires=3, inverse=False)
        density=tq.DensityMatrix(n_wires=4,bsz=1)
        density.init_by_pure_state(state)
        density.hadamard(wires=2, inverse=False)
        density.hadamard(wires=3, inverse=False)
        self.assertTrue(np.allclose(density.matrices, all_zero_state(4)))

    def test_einsum(self):
        return
    
    def test_bmm(self):
        return



if __name__ == '__main__':    
    unittest.main()

