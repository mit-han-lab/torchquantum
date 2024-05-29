import torchquantum as tq
import numpy as np

import qiskit.circuit.library.standard_gates as qiskit_gate
from qiskit.quantum_info import DensityMatrix as qiskitDensity

from unittest import TestCase
from random import randrange

maximum_qubit_num = 5

single_gate_list = [
    {"qiskit": qiskit_gate.HGate, "tq": tq.h, "name": "Hadamard"},
    {"qiskit": qiskit_gate.XGate, "tq": tq.x, "name": "x"},
    # {"qiskit": qiskit_gate.YGate, "tq": tq.y, "name": "y"},
    {"qiskit": qiskit_gate.ZGate, "tq": tq.z, "name": "z"},
    {"qiskit": qiskit_gate.SGate, "tq": tq.S, "name": "S"},
    {"qiskit": qiskit_gate.TGate, "tq": tq.T, "name": "T"},
    # {"qiskit": qiskit_gate.SXGate, "tq": tq.SX, "name": "SX"},
    {"qiskit": qiskit_gate.SdgGate, "tq": tq.SDG, "name": "SDG"},
    {"qiskit": qiskit_gate.TdgGate, "tq": tq.TDG, "name": "TDG"}
]

single_param_gate_list = [

]

two_qubit_gate_list = [
    {"qiskit": qiskit_gate.CXGate, "tq": tq.CNOT, "name": "CNOT"},
    {"qiskit": qiskit_gate.CYGate, "tq": tq.CY, "name": "CY"},
    {"qiskit": qiskit_gate.CZGate, "tq": tq.CZ, "name": "CZ"},
    {"qiskit": qiskit_gate.SwapGate, "tq": tq.SWAP, "name": "SWAP"}
]

two_qubit_param_gate_list = [

]

three_qubit_gate_list = [
    {"qiskit": qiskit_gate.CCXGate, "tq": tq.Toffoli, "name": "Toffoli"},
    {"qiskit": qiskit_gate.CSwapGate, "tq": tq.CSWAP, "name": "CSWAP"}
]

three_qubit_param_gate_list = [
]


class trace_preserving_test(TestCase):

    def mix_random_layer_trace(self, gatestrength):
        passed = True
        three_qubit_gate_length = len(three_qubit_gate_list)
        single_qubit_gate_length = len(single_gate_list)
        two_qubit_gate_length = len(two_qubit_gate_list)

        for qubit_num in range(3, maximum_qubit_num + 1):
            qdev = tq.NoiseDevice(n_wires=qubit_num, bsz=1, device="cpu", record_op=True)
            gate_num = int(gatestrength * qubit_num)
            for i in range(0, gate_num + 1):
                random_gate_qubit_num = randrange(3)
                '''
                Add a single qubit gate
                '''
                if (random_gate_qubit_num == 0):
                    random_gate_index = randrange(single_qubit_gate_length)
                    gate_pair = single_gate_list[random_gate_index]
                    random_qubit_index = randrange(qubit_num)
                    gate_pair['tq'](qdev, [random_qubit_index])

                '''
                Add a two qubit gate
                '''
                if (random_gate_qubit_num == 1):
                    random_gate_index = randrange(two_qubit_gate_length)
                    gate_pair = two_qubit_gate_list[random_gate_index]
                    random_qubit_index1 = randrange(qubit_num)
                    random_qubit_index2 = randrange(qubit_num)
                    while random_qubit_index2 == random_qubit_index1:
                        random_qubit_index2 = randrange(qubit_num)
                    gate_pair['tq'](qdev, [random_qubit_index1, random_qubit_index2])
                '''
                Add a three qubit gate
                '''
                if (random_gate_qubit_num == 2):
                    random_gate_index = randrange(three_qubit_gate_length)
                    gate_pair = three_qubit_gate_list[random_gate_index]
                    random_qubit_index1 = randrange(qubit_num)
                    random_qubit_index2 = randrange(qubit_num)
                    while random_qubit_index2 == random_qubit_index1:
                        random_qubit_index2 = randrange(qubit_num)
                    random_qubit_index3 = randrange(qubit_num)
                    while random_qubit_index3 == random_qubit_index1 or random_qubit_index3 == random_qubit_index2:
                        random_qubit_index3 = randrange(qubit_num)
                    gate_pair['tq'](qdev, [random_qubit_index1, random_qubit_index2, random_qubit_index3])

            if not np.isclose(qdev.calc_trace(0), 1):
                passed = False
                print("Trace not preserved: %f" % (qdev.calc_trace(0)))
            else:
                print("Trace preserved: %f" % (qdev.calc_trace(0)))
        return passed

    def test_mix_random_layer_trace(self):
        repeat_num = 5
        gate_strength_list = [0.5, 1, 1.5, 2]
        for i in range(0, repeat_num):
            for gatestrength in gate_strength_list:
                self.assertTrue(self.mix_random_layer_trace(gatestrength))
