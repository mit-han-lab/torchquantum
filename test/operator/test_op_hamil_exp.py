from torchquantum.operator import OpHamilExp
from torchquantum.algorithm import Hamiltonian
import numpy as np
from test.utils import check_all_close
from torchquantum.device import QuantumDevice

def test_op_hamil_exp():
    hamil = Hamiltonian(coeffs=[1.0, 0.5], paulis=['ZZ', 'XX'])
    op = OpHamilExp(hamil=hamil,
                    trainable=True,
                    theta=0.45)
    
    print(op.matrix)
    print(op.exponent_matrix)

    check_all_close(
        op.matrix,
        np.array([[ 0.9686-0.2217j,  0.0000+0.0000j,  0.0000+0.0000j, -0.0250-0.1094j],
        [ 0.0000+0.0000j,  0.9686+0.2217j,  0.0250-0.1094j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0250-0.1094j,  0.9686+0.2217j,  0.0000+0.0000j],
        [-0.0250-0.1094j,  0.0000+0.0000j,  0.0000+0.0000j,  0.9686-0.2217j]])
    )

    check_all_close(
        op.exponent_matrix,
        np.array([[0.-0.2250j, 0.+0.0000j, 0.+0.0000j, 0.-0.1125j],
        [0.+0.0000j, 0.+0.2250j, 0.-0.1125j, 0.+0.0000j],
        [0.+0.0000j, 0.-0.1125j, 0.+0.2250j, 0.+0.0000j],
        [0.-0.1125j, 0.+0.0000j, 0.+0.0000j, 0.-0.2250j]])
    )

    qdev = QuantumDevice(n_wires=2)
    qdev.reset_states(bsz=2)

    op(qdev, wires=[1, 0])

    print(qdev.get_states_1d().cpu().detach().numpy())

    check_all_close(
        qdev.get_states_1d().cpu().detach().numpy(),
        np.array([[ 0.9686322 -0.22169423j , 0.        +0.j        ,  0.        +0.j, -0.02504631-0.1094314j ],
                  [ 0.9686322 -0.22169423j , 0.        +0.j        ,  0.        +0.j, -0.02504631-0.1094314j ]])
    )

if __name__ == '__main__':
    # import pdb
    # pdb.set_trace()
    test_op_hamil_exp()
