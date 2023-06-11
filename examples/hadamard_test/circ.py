import torchquantum as tq
import torch

def create_circ():
    qdev = tq.QuantumDevice(n_wires=4)

    hamil1 = tq.algorithm.Hamiltonian(coeffs=[1], paulis=["ZZX"])
    hamil2 = tq.algorithm.Hamiltonian(coeffs=[1, 0.5], paulis=["ZZZ", "XXX"])
    op_hamil_exp1 = tq.operator.OpHamilExp(hamil1, trainable=True, theta=0.5)
    op_hamil_exp2 = tq.operator.OpHamilExp(hamil2, trainable=True, theta=0.5)
    qdev.paulix(wires=0)
    qdev.controlled_unitary(params=op_hamil_exp1.matrix, c_wires=[0], t_wires=[1, 2, 3])
    qdev.controlled_unitary(params=op_hamil_exp2.matrix, c_wires=[0], t_wires=[1, 2, 3])

    print(tq.measurement.measure(qdev, n_shots=1024))

class Circ(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        hamil1 = tq.algorithm.Hamiltonian(coeffs=[1], paulis=["ZZX"])
        hamil2 = tq.algorithm.Hamiltonian(coeffs=[1, 0.5], paulis=["ZZZ", "XXX"])
        self.op_hamil_exp1 = tq.operator.OpHamilExp(hamil1, trainable=True, theta=0.5)
        self.op_hamil_exp2 = tq.operator.OpHamilExp(hamil2, trainable=True, theta=0.5)

    def forward(self):
        qdev = tq.QuantumDevice(n_wires=4)
        qdev.paulix(wires=0)
        qdev.controlled_unitary(params=self.op_hamil_exp1.matrix, c_wires=[0], t_wires=[1, 2, 3])
        qdev.controlled_unitary(params=self.op_hamil_exp2.matrix, c_wires=[0], t_wires=[1, 2, 3])
        return tq.measurement.expval_joint_analytical(qdev, "ZZZZ")

if __name__ == '__main__':
    # import pdb
    # pdb.set_trace()

    # method 1
    create_circ()


    # method 2
    circ = Circ()
    expv = circ()
    print(expv)
    optimizer = torch.optim.Adam(circ.parameters(), lr=0.001)
    #### update the two thetas to make expv close to 0.5
    for step in range(2000):
        optimizer.zero_grad()
        expv = circ()
        loss = torch.abs(expv - 0.5)
        loss.backward()
        optimizer.step()
        print("step: {}, loss: {}".format(step, loss))
    print(f"theta1: {circ.op_hamil_exp1.theta}")
    print(f"theta2: {circ.op_hamil_exp2.theta}")
