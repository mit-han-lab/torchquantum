import torchquantum as tq

__all__ = ["Grover"]

class GroverResult(object):
	"""Result class for Grover algorithm"""
	def __init__(self) -> None:
			self.iterations: int

class Grover(object):
	"""Grover's search algorithm based the paper "A fast quantum mechanical algorithm for database search" by Lov K. Grover
	https://arxiv.org/abs/quant-ph/9605043
	"""
		
	def __init__(self, oracle: tq.module.QuantumModule, iterations: int, n_wires:int) -> None:
		"""
		Args:
			oracle (tq.module.QuantumModule): The oracle is a quantum module that adds a negative phase to the 
						solution states.
			iterations (int): The number of iterations to run the algorithm for.
			n_wires (int): The number of qubits used in the quantum circuit.
		"""
		super().__init__()
		self._oracle = oracle
		self._iterations = iterations
		self._n_wires = n_wires
		
	def initial_state_prep(self):
		"""
		Prepares the initial state of a quantum circuit by applying a Hadamard gate to each qubit.
		
		Returns:
			a `QuantumModule` object that represents the initial state preparation circuit.
		"""
		ops = []
		for i in range(self._n_wires):
			ops.append({'name': 'hadamard', 'wires': i})
		return tq.QuantumModule.from_op_history(ops)
	
	def diffusion_operator(self):
		"""
		Prepares the diffusion operator for the grover's circuit.
	
		Returns:
			a quantum module that represents the diffusion operator for a quantum circuit.
		"""
		ops = []
		hadamards = [{'name': 'hadamard', 'wires': i} for i in range(self._n_wires)]
		flips = [{'name': 'x', 'wires': i} for i in range(self._n_wires)]
		
		ops += hadamards
		ops += flips
		
		if self._n_wires  == 1:
				ops += [{'name': 'z', 'wires': 0}]
		else:
				ops += [{'name': 'hadamard', 'wires': self._n_wires - 1}]
				ops += [{'name': 'multicnot', 'n_wires': self._n_wires, 'wires': range(self._n_wires)}]
				ops += [{'name': 'hadamard', 'wires': self._n_wires - 1}]
		
		ops += flips
		ops += hadamards
		
		return tq.QuantumModule.from_op_history(ops)
			
					
	def construct_grover_circuit(self, qdev: tq.QuantumDevice):
		"""
		Constructs a Grover's algorithm circuit with an initial state preparation, oracle,
		and diffusion operator, and iterates through them a specified number of times.
		
		Args:
			qdev (tq.QuantumDevice): tq.QuantumDevice is an object representing a quantum device or
		simulator on which quantum circuits can be executed. 
				
		Returns:
			the modified quantum device `qdev` after applying the Grover's algorithm circuit with the
		specified number of iterations.
		"""

		self.initial_state_prep()(qdev)        
		for _ in range(self._iterations):
			self._oracle(qdev)
			self.diffusion_operator()(qdev)
		
		return qdev
	
	def execute(self, qdev: tq.QuantumDevice, n_shots: int =1024):
		"""
		Executes a Grover search algorithm on a given quantum device and returns the result.
		
		Args:
			qdev (tq.QuantumDevice): tq.QuantumDevice is an object representing a quantum device or
		simulator on which quantum circuits can be executed. 
			n_shots (int): The number of times the circuit is run to obtain measurement statistics.
		Defaults to 1024
		
		Returns:
			an instance of the `GroverResult` class, which contains information about the results of
		running the Grover search algorithm on a quantum device. The `GroverResult` object includes the
		number of iterations performed, the measured bitstring, the top measurement (i.e. the most
		frequently measured bitstring), and the maximum probability of measuring the top measurement.
		"""
			
		qdev = self.construct_grover_circuit(qdev)
		bitstring = tq.measure(qdev, n_shots=n_shots)
		top_measurement, max_probability = max(bitstring[0].items(), key=lambda x: x[1])
		max_probability /= n_shots
		
		result = GroverResult()
		result.iterations = self._iterations
		result.bitstring = bitstring
		result.top_measurement = top_measurement
		result.max_probability = max_probability

		return result