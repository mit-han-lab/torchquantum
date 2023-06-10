import torch
import torchquantum as tq
import numpy as np


def test_func_mat_exp():
    qdev = tq.QuantumDevice(n_wires=3)
    qdev.reset_states(bsz=1)

    qdev.matrix_exp(wires=[0], params=torch.tensor([[1., 2.], [3., 4.+1.j]]))

    assert np.allclose(
        qdev.get_states_1d().cpu().detach().numpy(), 
        np.array(
        [[44.2796+23.9129j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j,
         85.5304+68.1896j,  0.0000+0.0000j,  0.0000+0.0000j,  0.0000+0.0000j]])
         )
    
    qdev = tq.QuantumDevice(n_wires=3)
    qdev.reset_states(bsz=2)

    qdev.matrix_exp(wires=[0, 2], params=torch.tensor([[1., 2., 2, 1], 
                                                       [3., 4.+1.j,  2, 1], 
                                                       [1., 2., 2, 1], 
                                                       [3., 4.+1.j,  2, 1]])
                                                       )  # type: ignore
    # print(qdev.get_states_1d().cpu().detach().numpy())

    assert np.allclose(
        qdev.get_states_1d().cpu().detach().numpy(), 
        np.array(
        [[483.20386+254.27155j, 747.27014+521.95013j,   0.+0.j, 0.+0.j, 482.2038+254.27151j, 747.27014+521.95013j, 0.+0.j,        0.+0.j],
         [483.20386+254.27155j, 747.27014+521.95013j,   0.+0.j, 0.+0.j, 482.2038+254.27151j, 747.27014+521.95013j, 0.+0.j,        0.+0.j]]
    ))

if __name__ == '__main__':
    import pdb
    pdb.set_trace()

    test_func_mat_exp()