import numpy as np
import torchquantum as tq


def test_non_trivial_pauli_expectation():
    class TestCircuit(tq.QuantumModule):
        def __init__(self):
            super().__init__()

            self.meas = tq.measurement.MeasureMultipleTimes([
                {'wires': range(2), 'observables': 'xx'},
                {'wires': range(2), 'observables': 'yy'},
                {'wires': range(2), 'observables': 'zz'},
            ])

        def forward(self, qdev: tq.QuantumDevice):
            """
                Prepare and measure the expexctation value of the state
                    exp(-i pi/8)/sqrt(2) * (cos pi/12,
                                            -i sin pi/12,
                                            -i sin pi/12 * exp(i pi/4),
                                            -i sin pi/12 * exp(i pi/4))
            """
            # prepare bell state
            tq.h(qdev, 0)
            tq.cnot(qdev, [0, 1])

            # add some phases
            tq.rz(qdev, wires=0, params=np.pi / 4)
            tq.rx(qdev, wires=1, params=np.pi / 6)
            return self.meas(qdev)

    test_circuit = TestCircuit()
    qdev = tq.QuantumDevice(bsz=1, n_wires=2)  # Batch size 1 for testing

    # Run the circuit
    meas_results = test_circuit(qdev)[0]

    # analytical results for XX, YY, ZZ expval respectively
    expval_xx = np.cos(np.pi / 4)
    expval_yy = -np.cos(np.pi / 4) * np.cos(np.pi / 6)
    expval_zz = np.cos(np.pi / 6)

    atol = 1e-6

    assert np.isclose(meas_results[0].item(), expval_xx, atol=atol), \
        f"Expected {expval_xx}, got {meas_results[0].item()}"
    assert np.isclose(meas_results[1].item(), expval_yy, atol=atol), \
        f"Expected {expval_yy}, got {meas_results[1].item()}"
    assert np.isclose(meas_results[2].item(), expval_zz, atol=atol), \
        f"Expected {expval_zz}, got {meas_results[2].item()}"

    print("Test passed!")


if __name__ == "__main__":
    test_non_trivial_pauli_expectation()
