import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.measurement import expval_joint_analytical

import pdb
pdb.set_trace()

x = tq.QuantumDevice(n_wires=4)

tqf.hadamard(x, wires=0)
tqf.x(x, wires=1)
tqf.cnot(x, wires=[0, 1])
tqf.hadamard(x, wires=2)
tqf.hadamard(x, wires=3)

# print the current state (dynamic computation graph supported)
print(x.get_states_1d())
# print(x.get_state_1d())


print(expval_joint_analytical(x, 'ZZZZ'))

