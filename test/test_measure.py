import torchquantum as tq
import pdb

if __name__ == '__main__':
    pdb.set_trace()
    state = tq.QuantumState(n_wires=3, bsz=2)
    state.x(wires=2)
    state.x(wires=1)

    # state.ry(wires=0, params=0.98)
    # state.rx(wires=1, params=1.2)
    # state.cnot(wires=[0, 2])
    bitstream = tq.measure(state, n_shots=1024, draw_id=1)
    print(bitstream)
