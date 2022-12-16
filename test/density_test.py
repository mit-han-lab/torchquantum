import torchquantum as tq
import pdb
import numpy as np
import torch



class Test(unittest.TestCase):
    def setUp(self):
        print('Density Matrix basic function test')

    def tearDown(self):
        print('Density Matrix basic function test')

    def test_initialization(self):
        return True
    
    def test_positive_semidefinite(self):
        return True
    
    def test_clone_matrix(self):
        return True
    
    def test_set_matrix(self):
        return True
    
    def test_partial_trace(self):
        return True



if __name__ == '__main__':
    density = tq.DensityMatrix(
        n_wires=4,
        bsz=1
    )
    density.print_2d(0)