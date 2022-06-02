import unittest

import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import torch


class Test(unittest.TestCase):
    def setUp(self):
        print('test start')

    def tearDown(self):
        print('test end')

    def test_Hadamard(self):
        q_device = tq.QuantumDevice(n_wires=1)
        q_device.reset_states(bsz=1)
        tqf.hadamard(q_device, wires=0)
        result = torch.tensor([[1. / np.sqrt(2), 1. / np.sqrt(2)]])
        self.assertTrue(np.allclose(result, q_device.states))

if __name__ == '__main__':    
    unittest.main()


# run $python test/op_unittest.py
# or  $python -m unittest test/op_unittest.py