from torchquantum.plugins import qiskit2tq_op_history
import torchquantum as tq
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit


def test_qiskit2tp_op_history():
    circ = QuantumCircuit(3, 3)
    circ.h(0)
    circ.rx(0.1, 1)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.u(0.1, 0.2, 0.3, 0)
    print(circ)
    ops = qiskit2tq_op_history(circ)
    qmodule = tq.QuantumModule.from_op_history(ops)
    print(qmodule.Operator_list)



if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    test_qiskit2tp_op_history()
