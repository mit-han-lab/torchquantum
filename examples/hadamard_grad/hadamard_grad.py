import torch
import torchquantum as tq
from torchquantum.measurement import expval_joint_analytical

def gradient_circuit(left, target, right, n_wires, observable):
    '''
    Compute the gradient for the target gate.

    Parameters:
    - left: The gates to the left of the target gate.
    - target: The target gate for which the gradient is computed.
    - right: The gates to the right of the target gate.
    - n_wires: The number of wires (quantum bits) in the circuit.
    - observable: The observable of the original circuit, for which the gradients are computed.

    Returns:
    - The computed gradient for the target gate, analytical
    '''
    ancilla_qubit = n_wires

    gradient = 0
    for coeff, pauli in zip(target['coeffs'], target['paulis']):

        dev = tq.QuantumDevice(n_wires=n_wires+1, bsz=1, device="cpu") # use device='cuda' for GPU
        # ancilla
        dev.h(wires=ancilla_qubit)
        # left
        for op_info in left:
            if 'coeffs' in op_info:
                # Evolution
                op = tq.operator.OpPauliExp(coeffs=op_info['coeffs'], paulis=op_info['paulis'], theta=op_info['params'], trainable=False)
                op(dev, wires=op_info['wires'])
            else:
                # other gates
                op = tq.QuantumModule.from_op_history([op_info])
                op(dev)
        # target
        generator = tq.algorithm.Hamiltonian(coeffs=[1.0], paulis=[pauli]) # ZZZZ
        dev.controlled_unitary(params=generator.matrix, c_wires=[ancilla_qubit], t_wires=target['wires'])
        # ancilla
        dev.h(wires=ancilla_qubit)
        # right
        for op_info in right:
            if 'coeffs' in op_info:
                # Evolution
                op = tq.operator.OpPauliExp(coeffs=op_info['coeffs'], paulis=op_info['paulis'], theta=op_info['params'], trainable=False)
                op(dev, wires=op_info['wires'])
            else:
                # other gates
                op = tq.QuantumModule.from_op_history([op_info])
                op(dev)

        # measurement
        original_measurement = observable # 'ZZZZ'
        expval = expval_joint_analytical(dev, original_measurement+'Y')
        gradient -= coeff * torch.mean(expval)

    return gradient

def hadamard_grad(op_history, n_wires, observable):
    '''
    Return the gradients for parameters in the q_device.

    Parameters:
    - op_history: The history of quantum operations applied on the q_device.
    - n_wires: The number of wires (quantum bits) in the q_device.
    - observable: The observable of the original circuit, for which the gradients are computed.

    Returns:
    - A list of gradients, ordered as the list the operations in op_history
    '''
    gradient_list = []
    for i, op in enumerate(op_history):
        if not op['trainable']:
            gradient_list.append(None)
        else:
            left = op_history[:i+1]
            target = op
            right = op_history[i+1:]
            gradient_list.append(
                gradient_circuit(left, target, right, n_wires, observable)
            )

    return gradient_list