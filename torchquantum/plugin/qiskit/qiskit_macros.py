"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torchquantum as tq

QISKIT_INCOMPATIBLE_OPS = [
    tq.Rot,
    tq.MultiRZ,
    tq.CRot,
]

QISKIT_INCOMPATIBLE_FUNC_NAMES = [
    "rot",
    "multirz",
    "crot",
]

IBMQ_NAMES = [
    "ibmqx2",
    "ibmq_16_melbourne",
    "ibmq_5_yorktown",
    "ibmq_armonk",
    "ibmq_athens",
    "ibmq_belem",
    "ibmq_bogota",
    "ibmq_casablanca",
    "ibmq_dublin",
    "ibmq_guadalupe",
    "ibmq_jakarta",
    "ibmq_kolkata",
    "ibmq_lima",
    "ibmq_manhattan",
    "ibmq_manila",
    "ibmq_montreal",
    "ibmq_mumbai",
    "ibmq_paris",
    "ibmq_qasm_simulator",
    "ibmq_quito",
    "ibmq_rome",
    "ibmq_santiago",
    "ibmq_sydney",
    "ibmq_toronto",
    "simulator_extended_stabilizer",
    "simulator_mps",
    "simulator_stabilizer",
    "simulator_statevector",
    "ibm_auckland",
    "ibm_cairo",
    "ibm_geneva",
    "ibm_hanoi",
    "ibm_ithaca",
    "ibm_lagos",
    "ibm_nairobi",
    "ibm_oslo",
    "ibm_peekskill",
    "ibm_perth",
    "ibm_washington",
]

IBMQ_PNAMES = [
    "FakeArmonk",
    "FakeBogota" "FakeQuito",
]
