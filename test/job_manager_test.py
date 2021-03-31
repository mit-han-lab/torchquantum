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
backend = provider.get_backend('ibmqx2')

# Build a thousand circuits.
circs = []
for _ in range(1000):
    circs.append(random_circuit(num_qubits=2, depth=3, measure=True))

# Need to transpile the circuits first.
circs = transpile(circs, backend=backend)

print('here1')

# Use Job Manager to break the circuits into multiple jobs.
job_manager = IBMQJobManager()
job_set_foo = job_manager.run(circs, backend=backend, name='foo')


print('here2')

results = job_set_foo.results()
results.get_counts(5)  # Counts for experiment 5.

