import numpy as np
import torchquantum as tq


def test_ch_applies_on_device():
    """CH must act as a Hadamard on the target when the control is |1>."""
    qdev = tq.QuantumDevice(n_wires=2)
    qdev.x(0)
    tq.CH()(qdev, wires=[0, 1])
    expected = np.array([0, 0, 1, 1]) / np.sqrt(2)
    np.testing.assert_allclose(qdev.get_states_1d()[0].numpy(), expected, atol=1e-6)


if __name__ == "__main__":
    test_ch_applies_on_device()
