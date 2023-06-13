import torchquantum as tq



class QVQEModel(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.gate1 = tq.operator.OpPauliExp(coeffs=[1.0], paulis=["ZZZZ"], theta=0.5, trainable=True)
        self.gate2 = tq.operator.OpPauliExp(coeffs=[1.0], paulis=["ZZII"], theta=0.5, trainable=True)
        self.gate3 = tq.operator.OpPauliExp(coeffs=[1.0], paulis=["ZZXX"], theta=0.5, trainable=True)
        self.gate4 = tq.operator.OpPauliExp(coeffs=[1.0], paulis=["YXIX"], theta=0.5, trainable=True)


    def forward(self):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=1, device='cpu', record_op=True
        )

        # self.gate1(qdev, wires=[0, 1, 2, 3])
        # self.gate2(qdev, wires=[0, 1, 2, 3])
        # self.gate3(qdev, wires=[0, 1, 2, 3])
        self.gate4(qdev, wires=[0, 1, 2, 3])

        expval = tq.measurement.expval_joint_analytical(qdev, observable="ZZZZ")

        return expval, qdev


model = QVQEModel()

expval, qdev = model()
print(expval)


print(qdev.op_history)


# mod = tq.QuantumModule.from_op_history(qdev.op_history)

# print(mod)

# for name, param in mod.named_parameters():
    # print(name, param)
