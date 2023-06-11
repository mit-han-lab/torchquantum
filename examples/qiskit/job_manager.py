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

from qiskit import IBMQ, transpile
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.circuit.random import random_circuit

import logging

logging.basicConfig(level=logging.DEBUG)

"""
WARNING: seems not work on server, reasons unknown yet
"""

# import pdb
# pdb.set_trace()

provider = IBMQ.load_account()
backend = provider.get_backend("ibmqx2")

# Build a thousand circuits.
circs = []
for _ in range(1000):
    circs.append(random_circuit(num_qubits=2, depth=3, measure=True))

# Need to transpile the circuits first.
circs = transpile(circs, backend=backend)

print("here1")

# Use Job Manager to break the circuits into multiple jobs.
job_manager = IBMQJobManager()
job_set_foo = job_manager.run(circs, backend=backend, name="foo")


print("here2")

results = job_set_foo.results()
results.get_counts(5)  # Counts for experiment 5.
