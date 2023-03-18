# import torch
# import torchquantum as tq
# import numpy as np
#
# from typing import Any
#
#
# class QuantizeFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx: Any, x: torch.Tensor) -> Any:
#         # should be round so that the changes would be small, values close to
#         # 2pi should go to 2pi
#         return x.round()
#
#     @staticmethod
#     def backward(ctx: Any, grad_output: Any) -> Any:
#         grad_input = grad_output.clone()
#         mean, std = grad_input.mean(), grad_input.std()
#         return grad_input.clamp_(mean - 3 * std, mean + 3 * std)
#
#
# class CliffordQuantizer(object):
#     def __init__(self):
#         pass
#
#     # straight-through estimator
#     @staticmethod
#     def quantize_sse(params):
#         param = params[0][0]
#         param = param % (2 * np.pi)
#         param = np.pi / 2 * QuantizeFunction.apply(param /
#                                                    (np.pi / 2))
#         params = param.unsqueeze(0).unsqueeze(0)
#         return params

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
