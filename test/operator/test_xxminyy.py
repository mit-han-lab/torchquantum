import torch
import torchquantum as tq


def test_xxminyy_applies_on_device():
    """XXMINYY must be applicable to a device and preserve the norm."""
    qdev = tq.QuantumDevice(n_wires=2)
    qdev.x(0)
    op = tq.XXMINYY(
        has_params=True, trainable=False, init_params=torch.tensor([[0.3, 0.5]])
    )
    op(qdev, wires=[0, 1])
    expected = op.matrix[:, 2]
    assert torch.allclose(qdev.get_states_1d()[0], expected, atol=1e-6)


if __name__ == "__main__":
    test_xxminyy_applies_on_device()
