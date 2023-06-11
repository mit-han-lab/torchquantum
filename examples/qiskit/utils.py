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

import qiskit

from qiskit import IBMQ
import pdb

pdb.set_trace()

IBMQ.load_account()

provider = IBMQ.get_provider(hub="ibm-q")
backend = provider.get_backend("ibmq_belem")

# print(backend.defaults().instruction_schedule_map._map)


# https://qiskit.org/documentation/tutorials/circuits_advanced/08_gathering_system_information.html
def describe_qubit(qubit, properties):
    """Print a string describing some of reported properties of the given qubit."""

    # Conversion factors from standard SI units
    us = 1e6
    ns = 1e9
    GHz = 1e-9

    print(
        "Qubit {0} has a \n"
        "  - T1 time of {1} microseconds\n"
        "  - T2 time of {2} microseconds\n"
        "  - U2 gate error of {3}\n"
        "  - U2 gate duration of {4} nanoseconds\n"
        "  - resonant frequency of {5} GHz".format(
            qubit,
            properties.t1(qubit) * us,
            properties.t2(qubit) * us,
            properties.gate_error("sx", qubit),
            properties.gate_length("sx", qubit) * ns,
            properties.frequency(qubit) * GHz,
        )
    )


props = backend.properties()
describe_qubit(0, props)


def get_2q_errors(props):
    """Print the 2-qubit gate fidelities for the given backend."""
    errors = {}
    for gate in props.gates:
        if len(gate.qubits) == 2:
            errors[gate.name] = gate.parameters[0]
    return errors


print(get_2q_errors(props))
