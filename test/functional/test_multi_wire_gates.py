import torch
import torchquantum as tq


def test_hadamard_multi_wires():
    # Multi-wire call
    q_dev_multi = tq.QuantumDevice(n_wires=3, bsz=2, device="cpu")
    q_dev_multi.h(wires=[0, 1, 2])

    # Sequential single-wire calls
    q_dev_seq = tq.QuantumDevice(n_wires=3, bsz=2, device="cpu")
    q_dev_seq.h(wires=0)
    q_dev_seq.h(wires=1)
    q_dev_seq.h(wires=2)

    assert torch.allclose(q_dev_multi.states, q_dev_seq.states), "Multi-wire Hadamard states must match sequential calls"


def test_pauli_x_multi_wires():
    q_dev_multi = tq.QuantumDevice(n_wires=2, bsz=1, device="cpu")
    q_dev_multi.x(wires=[0, 1])

    q_dev_seq = tq.QuantumDevice(n_wires=2, bsz=1, device="cpu")
    q_dev_seq.x(wires=0)
    q_dev_seq.x(wires=1)

    assert torch.allclose(q_dev_multi.states, q_dev_seq.states), "Multi-wire PauliX states must match sequential calls"


def test_two_qubit_gate_remains_unaffected():
    q_dev = tq.QuantumDevice(n_wires=2, bsz=1, device="cpu")
    q_dev.h(wires=0)
    q_dev.cnot(wires=[0, 1])
    # Bell state (|00> + |11>) / sqrt(2)
    expected_density_state = torch.zeros(1, 2, 2, dtype=tq.C_DTYPE)
    expected_density_state[0, 0, 0] = 1.0 / 2**0.5
    expected_density_state[0, 1, 1] = 1.0 / 2**0.5
    assert torch.allclose(q_dev.states, expected_density_state)
