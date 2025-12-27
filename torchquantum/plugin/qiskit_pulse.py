import torch
import torchquantum as tq
# from qiskit import pulse, QuantumCircuit
from qiskit import QuantumCircuit, transpile, pulse
# from qiskit.pulse import library
from qiskit.pulse import Schedule, InstructionScheduleMap
from qiskit_ibm_provider.fake_provider import FakeQuitoV2, FakeArmonkV2, FakeBogotaV2
# rom qiskit.test.mock import FakeQuito, FakeArmonk, FakeBogota
# from qiskit.compiler import assemble, schedule
# from .qiskit_macros import IBMQ_PNAMES
from qiskit.transpiler import PassManager, preset_passmanagers


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

    # Initialize the fake backend
    # backend = name()
    # Add measurement to circuit if needed
    return
