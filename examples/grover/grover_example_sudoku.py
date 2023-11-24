""" 
This example is based on Qiskt's Texbook: https://learn.qiskit.org/course/ch-algorithms/grovers-algorithm#sudoku

We will now tackle a 2x2 binary sudoku problem using Grover's algorithm,  where we don't necessarily possess 
prior knowledge of the solution. The problem adheres to two simple rules:

1. No column can have the same value repeated.
2. No row can have the same value repeated.

We will use the following 4 variables:

---------
| a | b |
---------
| c | d |
---------

Please keep in mind that while utilizing Grover's algorithm to solve this particular problem may not be practical 
(as the solution can likely be determined mentally), the intention of this example is to showcase the process of 
transforming classical decision problems into oracles suitable for Grover's algorithm.

We need to check for four conditions:

1. a != b
2. c != d
3. a != c
4. b != d
"""

import torchquantum as tq
from torchquantum.algorithm import Grover


# To simplify the process, we can compile this set of comparisons into a list of clauses for convenience.
clauses = [ [0, 1], [0, 2], [1, 3], [2, 3] ]

# This circuit checks if input0 is equal to input1 and stores the output in output. 
# The output of each comparison is stored in a new bit.
def XOR(input0, input1, output):
    op1 = {'name': 'cnot', 'wires': [input0, output]}
    op2 = {'name': 'cnot', 'wires': [input1, output]}
    return [op1, op2]
    
# To verify each clause, we repeat the above circuit for every pairing in the `clauses`. 
ops = []
clause_qubits = [4, 5, 6, 7]
for i, clause in enumerate(clauses):
	ops += XOR(clause[0], clause[1], clause_qubits[i])

# To determine if the assignments of a, b, c, d are a solution to the sudoku, we examine the final state 
# of the `clause_qubits`. Only when all of these qubits are 1, it indicates that the clauses are satisfied. 
# To achieve this, we incorporate a multi-controlled Toffoli gate in our checking circuit. This gate 
# ensures that a single output bit will be set to 1 if and only if all the clauses are satisfied, 
# allowing us to easily determine if our assignment is a solution.
ops += [{'name': 'multicnot', 'n_wires': 5, 'wires': [4,5,6,7,8]}]

# In order to transform our checking circuit into a Grover oracle, it is crucial to ensure that the `clause_qubits` 
# are always returned to their initial state after the computation. This guarantees that `clause_qubits` are all 
# set to 0 once our circuit has finished running. To achieve this, we include a step called "uncomputation" 
# where we repeat the segment of the circuit that computes the clauses. This uncomputation step ensures the 
# desired state restoration, enabling us to effectively use the circuit as a Grover oracle.
for qubit, clause in enumerate(clauses):
	ops += XOR(clause[0], clause[1], qubit + 4)

# Full Algorithm
# We can combine all the components we have discussed so far

qmodule = tq.QuantumModule.from_op_history(ops)
iterations = 2
qdev = tq.QuantumDevice(n_wires=9, device="cpu")

# Initialize output qubit (last qubit) in state |->
qdev.x(wires=8)
qdev.h(wires=8)

# Perform Grover's Search
grover = Grover(qmodule, iterations, 4)
result = grover.execute(qdev)
bitstring = result.bitstring[0]

# Extract the top two most likely solutions
res = {k: v for k, v in sorted(bitstring.items(), key=lambda item: item[1], reverse=True)}

# Print the top two most likely solutions
top_keys = list(res.keys())[:2]
print("Top two most likely solutions:")
for key in top_keys:
	print("Solution: ", key)
	print("a = ", key[0])
	print("b = ", key[1])
	print("c = ", key[2])
	print("d = ", key[3])
	print("")
