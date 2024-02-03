import torchquantum as tq
import numpy as np

import qiskit.circuit.library.standard_gates as qiskit_gate
from qiskit.quantum_info import DensityMatrix as qiskitDensity

from unittest import TestCase





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



class trace_test(TestCase):
    def test_single_qubit_trace_preserving(self):
        return

    def test_two_qubit_trace_preserving(self):
        return


    def test_three_qubit_trace_preserving(self):
        return

