import torchquantum as tq
import pdb
import numpy as np
import torch

if __name__ == '__main__':
    pdb.set_trace()
    state = tq.QuantumState(
        n_wires=4,
        bsz=1
    )
    state.hadamard(wires=1, inverse=True)
    state.h(wires=2)
    state.rx(wires=1, params=torch.Tensor([np.pi]))

    state.h(wires=0)
    state.rx(wires=1, params=0.1)

    state.u1(wires=1, params=[0.2])
    state.qubitunitary(wires=0, params=[[0,1], [1,0]])

    state.u3(wires=2, params=[0.1, 0.2, 0.3])
    state.crot(wires=[0,3], params=[0.3, 0.4, 0.5])
    state.rot(wires=3, params=[0.4, 0.5, 0.6])

    print(state)


