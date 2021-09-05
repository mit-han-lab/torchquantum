from qiskit import QuantumCircuit, assemble, Aer
from qiskit import IBMQ

IBMQ.save_account('51a2a5d55d3e1d9683ab4f135fe6fbb84ecf3221765e19adb408699d43c6eaa238265059c3c2955ba59328634ffbd88ba14d5386c947d22eb9a826e40811d626', overwrite=True)
IBMQ.load_account()

be = IBMQ.providers()[2].get_backend('ibmq_guadalupe')
qc = QuantumCircuit(1, 1)
qc.x(0)
qc.measure(0, 0)
print(qc)
qobj = be.run(qc)
print('start run')
result = qobj.result()
print('end run')
print(result)
