import torch
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit.circuit.library.standard_gates as qiskit_gate

from qiskit import QuantumCircuit
from qiskit import Aer, execute
from torchpack.utils.logging import logger
from torchquantum.utils import switch_little_big_endian_matrix


__all__ = ['tq2qiskit']


# construct a QuantumCircuit object according to the tq module
def tq2qiskit(m: tq.QuantumModule, x=None, draw=False):
    # build the module list without changing the statevector of QuantumDevice
    original_wires_per_block = m.wires_per_block
    original_static_mode = m.static_mode
    m.static_off()
    m.static_on(wires_per_block=m.q_device.n_wires)
    m.is_graph_top = False

    # forward to register all modules and parameters
    if x is None:
        m.forward(m.q_device)
    else:
        m.forward(m.q_device, x)

    m.is_graph_top = True
    m.graph.build_flat_module_list()

    module_list = m.graph.flat_module_list
    m.static_off()

    if original_static_mode:
        m.static_on(wires_per_block=original_wires_per_block)

    circ = QuantumCircuit(m.q_device.n_wires, m.q_device.n_wires)

    for module in module_list:
        try:
            # no params in module or batch size == 1, because we will
            # generate only one qiskit QuantumCircuit
            assert (module.params is None or module.params.shape[0] == 1)
        except AssertionError:
            logger.exception(f"Cannot convert batch model tq module")

    for module in module_list:
        if module.name == 'Hadamard':
            circ.h(*module.wires)
        elif module.name == 'PauliX':
            circ.x(*module.wires)
        elif module.name == 'PauliY':
            circ.y(*module.wires)
        elif module.name == 'PauliZ':
            circ.z(*module.wires)
        elif module.name == 'S':
            circ.s(*module.wires)
        elif module.name == 'T':
            circ.t(*module.wires)
        elif module.name == 'SX':
            circ.sx(*module.wires)
        elif module.name == 'CNOT':
            circ.cnot(*module.wires)
        elif module.name == 'CZ':
            circ.cz(*module.wires)
        elif module.name == 'CY':
            circ.cy(*module.wires)
        elif module.name == 'RX':
            circ.rx(module.params[0][0].item(), *module.wires)
        elif module.name == 'RY':
            circ.ry(module.params[0][0].item(), *module.wires)
        elif module.name == 'RZ':
            circ.rz(module.params[0][0].item(), *module.wires)
        elif module.name == 'SWAP':
            circ.swap(*module.wires)
        elif module.name == 'CSWAP':
            circ.cswap(*module.wires)
        elif module.name == 'Toffoli':
            circ.ccx(*module.wires)
        elif module.name == 'PhaseShift':
            circ.p(module.params[0][0].item(), *module.wires)
        elif module.name == 'CRX':
            circ.crx(module.params[0][0].item(), *module.wires)
        elif module.name == 'CRY':
            circ.cry(module.params[0][0].item(), *module.wires)
        elif module.name == 'CRZ':
            circ.crz(module.params[0][0].item(), *module.wires)
        elif module.name == 'U1':
            circ.u1(module.params[0][0].item(), *module.wires)
        elif module.name == 'U2':
            circ.u2(*list(module.params[0].data.numpy()), *module.wires)
        elif module.name == 'U3':
            circ.u3(*list(module.params[0].data.numpy()), *module.wires)
        elif module.name == 'QubitUnitary' or \
                module.name == 'QubitUnitaryFast' or \
                module.name == 'TrainableUnitary' or \
                module.name == 'TrainableUnitaryStrict':
            from torchquantum.plugins.qiskit_unitary_gate import UnitaryGate
            mat = module.params[0].data.numpy()
            mat = switch_little_big_endian_matrix(mat)
            circ.append(UnitaryGate(mat), module.wires, [])
        elif module.name == 'MultiCNOT':
            circ.mcx(module.wires[:-1], module.wires[-1])
        elif module.name == 'MultiXCNOT':
            controls = module.wires[:-1]
            target = module.wires[-1]
            num_ctrl_qubits = len(controls)

            gate = qiskit_gate.MCXGrayCode(num_ctrl_qubits,
                                           ctrl_state='0' * num_ctrl_qubits)
            circ.append(gate, controls + [target], [])
        else:
            logger.exception(f"{module.name} cannot be converted to Qiskit.")
            raise NotImplementedError(module.name)

        if module.inverse:
            data = list(circ.data[-1])
            del circ.data[-1]
            circ.data.append(tuple([data[0].inverse()] + data[1:]))
    if draw:
        import matplotlib.pyplot as plt
        circ.draw()
        plt.show()
    return circ


class T00(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.gate = tq.Hadamard()

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        self.gate(q_device, wires=0)


class TQAll(tq.QuantumModule):
    def __init__(self, n_gate: int, op: tq.Operator):
        super().__init__()
        self.submodules = tq.QuantumModuleList()
        self.n_gate = n_gate
        self.t00 = T00()
        for k in range(self.n_gate):
            self.submodules.append(op())

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        self.q_device = q_device
        for k in range(self.n_gate - 1):
            self.submodules[k](q_device, wires=[k, k + 1])
        self.submodules[-1](q_device, wires=[self.n_gate - 1, 0])
        self.t00(q_device)


class TestModule(tq.QuantumModule):
    def __init__(self, q_device: tq.QuantumDevice = None):
        super().__init__()
        self.q_device = q_device
        self.n_gate = 10
        self.gate0 = tq.CNOT()
        # self.gate1 = tq.CNOT()
        self.submodules = tq.QuantumModuleList()
        self.q_layer0 = TQAll(self.n_gate, tq.CNOT)
        for k in range(self.n_gate):
            self.submodules.append(tq.RY())
        # for k in range(self.n_gate):
        #     self.submodules.append(tq.CNOT())
        # self.gate0 = tq.RY(has_params=False, trainable=False)
        # self.gate1 = tq.RX(has_params=False, trainable=False)
        # self.gate2 = tq.RZ(has_params=False, trainable=False)
        self.gate1 = tq.RX(has_params=True, trainable=True)
        self.gate2 = tq.RZ(has_params=True, trainable=True)
        self.gate3 = tq.RY(has_params=True, trainable=True)
        # self.gate3 = tq.CNOT()
        self.gate4 = tq.RX(has_params=True, trainable=True)
        self.gate5 = tq.RZ(has_params=True, trainable=True)
        self.gate6 = tq.RY(has_params=True, trainable=True)
        self.gate7 = tq.RX()
        self.gate8 = tq.U2(has_params=True, trainable=True)
        self.gate9 = tq.TrainableUnitary(has_params=True, trainable=True,
                                         n_wires=3)
        self.gate10 = tq.MultiXCNOT(n_wires=5)
        self.gate11 = tq.MultiCNOT(n_wires=3)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        self.gate1(q_device, wires=3)
        self.gate2(q_device, wires=4)
        self.gate3(q_device, wires=3)
        self.gate4(q_device, wires=3)
        self.gate5(q_device, wires=3)
        self.gate6(q_device, wires=3, inverse=True)
        self.gate7(q_device, wires=4, params=x, inverse=True)
        self.gate8(q_device, wires=2)
        self.gate9(q_device, wires=[2, 3, 4])

        self.q_layer0(q_device)
        tqf.qubitunitary(self.q_device, wires=[1, 2], params=[[1, 0, 0, 0],
                                                              [0, 1, 0, 0],
                                                              [0, 0, 0, 1],
                                                              [0, 0, 1, 0]],
                         static=self.static_mode, parent_graph=self.graph)
        tqf.qubitunitary(self.q_device, wires=[1, 2], params=[[0, 1, 0, 0],
                                                              [1, 0, 0, 0],
                                                              [0, 0, 1, 0],
                                                              [0, 0, 0, 1]],
                         static=self.static_mode, parent_graph=self.graph)
        self.gate10(q_device, wires=[4, 5, 6, 7, 1])
        self.gate11(q_device, wires=[2, 1, 9])

        # self.gate0(q_device, wires=[7, 4])
        # self.gate1(q_device, wires=[3, 9])

        # self.gate0(q_device, wires=1, params=x[:, 2])
        # self.gate1(q_device, wires=5, params=x[:, 0])
        # self.gate2(q_device, wires=7, params=x[:, 6])

        # self.gate2(q_device, wires=5)
        # self.gate3(q_device, wires=[3, 5])
        # self.gate4(q_device, wires=5)


if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    inputs = torch.ones((1, 1)) * 0.42
    q_dev = tq.QuantumDevice(n_wires=10)
    test_module = TestModule(q_dev)

    circuit = tq2qiskit(test_module, inputs)

    simulator = Aer.get_backend('unitary_simulator')
    result = execute(circuit, simulator).result()
    unitary_qiskit = result.get_unitary(circuit)

    unitary_tq = test_module.get_unitary(q_dev, inputs)
    unitary_tq = switch_little_big_endian_matrix(unitary_tq.data.numpy())

    print(unitary_qiskit)
    print(unitary_tq)
