import torchquantum as tq
import torch
from torchquantum.macro import C_DTYPE, C_DTYPE_NUMPY

import numpy as np


def test_state_init():
    qdev = tq.QuantumDevice(n_wires=2, bsz=2)
    np.testing.assert_array_equal(
        qdev.get_states_1d().cpu().data.numpy(),
        np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=C_DTYPE_NUMPY),
    )

    qdev = tq.QuantumDevice(n_wires=2, bsz=1)
    np.testing.assert_array_equal(
        qdev.get_states_1d().cpu().data.numpy(),
        np.array([[1, 0, 0, 0]], dtype=C_DTYPE_NUMPY),
    )
