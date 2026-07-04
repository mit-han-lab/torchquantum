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

"""VQE for the H2 molecule with qiskit (>=2.x) V2 primitives.

A pure-qiskit reference example, rewritten from the legacy
qiskit.algorithms.VQE / opflow / QuantumInstance stack that was removed
from qiskit. The variational loop is StatevectorEstimator + scipy.
"""

import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

# H2 at 0.735 Angstrom, STO-3G, parity-mapped to 2 qubits
# (includes the nuclear repulsion offset on the identity term)
hamiltonian = SparsePauliOp.from_list(
    [
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ]
)

ansatz = efficient_su2(hamiltonian.num_qubits, reps=2)
estimator = StatevectorEstimator()


def cost(params):
    pub = (ansatz, hamiltonian, params)
    result = estimator.run([pub]).result()[0]
    return float(result.data.evs)


rng = np.random.default_rng(seed=42)
x0 = 2 * np.pi * rng.random(ansatz.num_parameters)

res = minimize(cost, x0, method="COBYLA", options={"maxiter": 500})

exact = float(np.min(np.linalg.eigvalsh(hamiltonian.to_matrix())))
print(f"VQE ground state energy:   {res.fun:.6f} Ha")
print(f"Exact ground state energy: {exact:.6f} Ha")
assert abs(res.fun - exact) < 1e-2, "VQE did not converge to the ground state"
