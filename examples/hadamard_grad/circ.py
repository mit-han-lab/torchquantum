import torchquantum as tq

op_types=(
    tq.Hadamard,
    tq.SHadamard,
    tq.PauliX,
    tq.PauliY,
    tq.PauliZ,
    tq.S,
    tq.T,
    tq.SX,
    tq.CNOT,
)

class Circ1(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.gate1 = tq.operator.OpPauliExp(coeffs=[1.0], paulis=["YXIX"], theta=0.5, trainable=True)

    def forward(self):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=1, device='cpu', record_op=True
        )
        self.gate1(qdev, wires=[0, 1, 2, 3])
        expval = tq.measurement.expval_joint_analytical(qdev, observable="ZZZZ")

        return expval, qdev

class Circ2(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.gate1 = tq.operator.OpPauliExp(coeffs=[1.0, 0.5], paulis=["YXIX", "ZIXZ"], theta=0.5, trainable=True)

    def forward(self):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=1, device='cpu', record_op=True
        )
        self.gate1(qdev, wires=[0, 1, 2, 3])
        expval = tq.measurement.expval_joint_analytical(qdev, observable="ZZZZ")

        return expval, qdev

class Circ3(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.gate1 = tq.operator.OpPauliExp(coeffs=[1.0], paulis=["XIII"], theta=0.5, trainable=True) # Note this is RX gate
        self.gate2 = tq.operator.OpPauliExp(coeffs=[0.722], paulis=["YXIX"], theta=0.5, trainable=True)
        self.gate3 = tq.operator.OpPauliExp(coeffs=[1.0, 0.5], paulis=["YXIX", "ZIXZ"], theta=0.2, trainable=True)
        self.gate4 = tq.operator.OpPauliExp(coeffs=[1.0], paulis=["YYYY"], theta=0.5, trainable=True)

        self.random_layer1 = tq.RandomLayer(op_types=op_types, n_ops=50, wires=list(range(self.n_wires)))
        self.random_layer2 = tq.RandomLayer(op_types=op_types, n_ops=50, wires=list(range(self.n_wires)))
        self.random_layer3 = tq.RandomLayer(op_types=op_types, n_ops=50, wires=list(range(self.n_wires)))
        self.random_layer4 = tq.RandomLayer(op_types=op_types, n_ops=50, wires=list(range(self.n_wires)))
        self.random_layer5 = tq.RandomLayer(op_types=op_types, n_ops=50, wires=list(range(self.n_wires)))


    def forward(self):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=1, device='cpu', record_op=True
        )

        qdev.h(wires=1)
        qdev.h(wires=3)
        qdev.cnot(wires=[1, 0])
        qdev.cnot(wires=[3, 2])

        self.random_layer1(qdev)
        self.gate1(qdev, wires=[0, 1, 2, 3])
        self.random_layer2(qdev)
        self.gate2(qdev, wires=[0, 1, 2, 3])
        self.random_layer3(qdev)
        self.gate3(qdev, wires=[0, 1, 2, 3])
        self.random_layer4(qdev)
        self.gate4(qdev, wires=[0, 1, 2, 3])
        self.random_layer5(qdev)

        expval = tq.measurement.expval_joint_analytical(qdev, observable="ZZZZ")

        return expval, qdev


if __name__ == '__main__':

    circ1 = Circ1()
    expval1, qdev1 = circ1()
    print('expval:')
    print(expval1)
    print('op_history:')
    print(qdev1.op_history)

    circ2 = Circ2()
    expval2, qdev2 = circ2()
    print('expval:')
    print(expval2)
    print('op_history:')
    print(qdev2.op_history)

    circ3 = Circ3()
    expval3, qdev3 = circ3()
    print('expval:')
    print(expval3)
    print('op_history:')
    print(qdev3.op_history)
