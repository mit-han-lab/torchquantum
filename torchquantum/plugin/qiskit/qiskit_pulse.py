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

# import torch
# import torchquantum as tq
# from qiskit import pulse, QuantumCircuit
# from qiskit import QuantumCircuit, transpile, pulse
# from qiskit.pulse import library
# from qiskit.pulse import Schedule, InstructionScheduleMap
# from qiskit_ibm_provider.fake_provider import FakeQuitoV2, FakeArmonkV2, FakeBogotaV2
# rom qiskit.test.mock import FakeQuito, FakeArmonk, FakeBogota
# from qiskit.compiler import assemble, schedule
# from .qiskit_macros import IBMQ_PNAMES
# from qiskit.transpiler import PassManager, preset_passmanagers


def circ2pulse(circuits, name):
    """
    Convert a circuit to a pulse schedule using the specified backend.

    Args:
        circuits (QuantumCircuit): The input quantum circuit.
        name (str): The name of the backend.

    Returns:
        None.

    Example:
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> circ2pulse(qc, 'ibmq_oslo')
    """

    """
    Old implementation:

    if name in IBMQ_PNAMES:
        backend = name()
        with pulse.build(backend) as pulse_tq:
            qc = circuits
            qc.measure_all()
            pulse.call(qc)
        pulse_tq.draw()
    """


    """
    The entire Qiskit Pulse package is being deprecated and will be moved to the Qiskit Dynamics repository.
    """

    return
