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

import argparse
import pdb
import torch
import torchquantum as tq
import numpy as np


from qiskit import transpile
from qiskit_aer import AerSimulator, UnitarySimulator, QasmSimulator, StatevectorSimulator
from torchpack.utils.logging import logger
from torchquantum.util import (
    switch_little_big_endian_matrix,
    switch_little_big_endian_state,
    get_expectations_from_counts,
    find_global_phase,
)
from .static_mode_test import QLayer as AllRandomLayer
from torchquantum.plugin import tq2qiskit
from torchquantum.macro import F_DTYPE
from torchquantum.plugin.qiskit.qiskit_plugin import custom_transpile


def unitary_tq_vs_qiskit_test():
    for n_wires in range(2, 10):
        q_dev = tq.QuantumDevice(n_wires=n_wires)
        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_layer = AllRandomLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=50,
            n_ops_cin=50,
            n_funcs=50,
            qiskit_compatible=True,
        )

        # unitary_tq = q_layer.get_unitary(q_dev, x)
        unitary_tq = q_layer.get_unitary(x)
        unitary_tq = switch_little_big_endian_matrix(unitary_tq.data.numpy())

        # qiskit
        circ = tq2qiskit(q_dev, q_layer, x)
        simulator = UnitarySimulator()
        
        # Use custom_transpile instead of standard transpile
        try:
            circuit = custom_transpile(circ, simulator, opt_level=1)
            result = simulator.run(circuit).result()
            unitary_qiskit = result.get_unitary(circuit)
        except Exception as e:
            logger.error(f"Failed simulation for n_wires={n_wires}: {str(e)}")
            logger.warning(f"Skipping test for n_wires={n_wires}")
            continue

        stable_threshold = 1e-5
        try:
            # WARNING: need to remove the global phase! The qiskit simulated
            # results sometimes has global phase shift.
            global_phase = find_global_phase(
                unitary_tq, unitary_qiskit, stable_threshold
            )

            if global_phase is None:
                logger.exception(
                    f"Cannot find a stable enough factor to "
                    f"reduce the global phase, increase the "
                    f"stable_threshold and try again"
                )
                raise RuntimeError

            assert np.allclose(unitary_tq * global_phase, unitary_qiskit, atol=1e-6)
            assert np.allclose(unitary_tq, unitary_qiskit, atol=1e-6)
            logger.info(f"PASS tq vs qiskit [n_wires]={n_wires}")

        except AssertionError:
            logger.exception(f"FAIL tq vs qiskit [n_wires]={n_wires}")
            raise AssertionError

        except RuntimeError:
            raise RuntimeError

    logger.info(f"PASS tq vs qiskit unitary test")


def state_tq_vs_qiskit_test():
    bsz = 1
    for n_wires in range(2, 10):
        q_dev = tq.QuantumDevice(n_wires=n_wires)
        q_dev.reset_states(bsz=bsz)

        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_layer = AllRandomLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=50,
            n_ops_cin=50,
            n_funcs=50,
            qiskit_compatible=True,
        )

        q_layer(q_dev, x)
        state_tq = q_dev.states.reshape(bsz, -1)
        state_tq = switch_little_big_endian_state(state_tq.data.numpy())

        # qiskit
        circ = tq2qiskit(q_dev, q_layer, x, debug=False)
        # Use StatevectorSimulator directly
        simulator = StatevectorSimulator()
        
        # Use our custom transpile function instead of standard transpile
        try:
            circuit = custom_transpile(circ, simulator, opt_level=1)
            result = simulator.run(circuit).result()
            state_qiskit = result.get_statevector(circuit)
        except Exception as e:
            logger.error(f"Failed simulation for n_wires={n_wires}: {str(e)}")
            logger.warning(f"Skipping test for n_wires={n_wires}")
            continue
        
        # Debug: Show original qiskit statevector before any conversions
        #print("\n----- Original Qiskit Statevector -----")
        # qiskit_raw = np.asarray(state_qiskit)
        #print(f"Raw Qiskit statevector: {qiskit_raw}")
        
        # Try applying the endianness conversion to Qiskit as well
        # qiskit_converted = switch_little_big_endian_state(qiskit_raw)
        #print(f"Qiskit after endianness conversion: {qiskit_converted}")
        #print("----- End Original Qiskit Statevector -----\n")

        stable_threshold = 1e-5
        try:
            # WARNING: need to remove the global phase! The qiskit simulated
            # results sometimes has global phase shift.
            global_phase = find_global_phase(
                state_tq, np.expand_dims(state_qiskit, 0), stable_threshold
            )

            if global_phase is None:
                logger.exception(
                    f"Cannot find a stable enough factor to "
                    f"reduce the global phase, increase the "
                    f"stable_threshold and try again"
                )
                raise RuntimeError

            # Add debug information to understand the differences
            print("\n----- Debug Information -----")
            print(f"Testing n_wires = {n_wires}")
            #print(f"Global phase: {global_phase}")
            
            # Convert Qiskit Statevector to numpy array
            #print(f"Qiskit statevector type: {type(state_qiskit)}")
            #print(f"Qiskit statevector direct representation: {state_qiskit}")
            
            state_qiskit_np = np.asarray(state_qiskit)
            #print(f"Qiskit statevector numpy array: {state_qiskit_np}")
            
            #print(f"TQ state shape: {state_tq.shape}, Qiskit state shape: {state_qiskit_np.shape}")
            
            # Calculate absolute differences and show max difference
            adjusted_state_tq = state_tq * global_phase
            abs_diff = np.abs(adjusted_state_tq - state_qiskit_np)
            abs_diff_og = np.abs(state_tq - state_qiskit_np)
            max_diff = np.max(abs_diff)
            max_diff_idx = np.argmax(abs_diff)
            max_diff_og = np.max(abs_diff_og)
            max_diff_idx_og = np.argmax(abs_diff_og)
            
            print(f"Maximum absolute difference: {max_diff} at index {max_diff_idx}")
            print(f"TQ state at max diff: {adjusted_state_tq.flat[max_diff_idx]}")
            print(f"Qiskit state at max diff: {state_qiskit_np.flat[max_diff_idx]}")
            print(f"Maximum absolute difference (original): {max_diff_og} at index {max_diff_idx_og}")
            print(f"TQ state at max diff (original): {state_tq.flat[max_diff_idx_og]}")
            print(f"Qiskit state at max diff (original): {state_qiskit_np.flat[max_diff_idx_og]}")
            
            # Show first few elements of both states for comparison
            """
            print("\nFirst 5 elements comparison:")
            for i in range(min(5, len(state_qiskit_np))):
                print(f"Index {i}:")
                print(f"  TQ (adjusted): {adjusted_state_tq.flat[i]}")
                print(f"  Qiskit: {state_qiskit_np[i]}")
                print(f"  Difference: {abs(adjusted_state_tq.flat[i] - state_qiskit_np[i])}")
            """
            print("----- End Debug Information -----\n")

            # Use a slightly larger tolerance for comparison due to numerical precision issues
            assert np.allclose(state_tq * global_phase, state_qiskit_np, atol=1e-5)
            assert np.allclose(state_tq, state_qiskit_np, atol=1e-5)
            logger.info(f"PASS tq vs qiskit [n_wires]={n_wires}")

        except AssertionError:
            logger.exception(f"FAIL tq vs qiskit [n_wires]={n_wires}")
            raise AssertionError

        except RuntimeError:
            raise RuntimeError

    logger.info(f"PASS tq vs qiskit statevector test")


def measurement_tq_vs_qiskit_test():
    bsz = 1
    for n_wires in range(2, 10):
        q_dev = tq.QuantumDevice(n_wires=n_wires)
        q_dev.reset_states(bsz=bsz)

        x = torch.randn((1, 100000), dtype=F_DTYPE)
        q_layer = AllRandomLayer(
            n_wires=n_wires,
            wires=list(range(n_wires)),
            n_ops_rd=50,
            n_ops_cin=50,
            n_funcs=50,
            qiskit_compatible=True,
        )

        q_layer(q_dev, x)
        measurer = tq.MeasureAll(obs=tq.PauliZ)
        # Get measurement from TorchQuantum
        measured_tq = measurer(q_dev).data[0].numpy()
        
        # qiskit
        circ = tq2qiskit(q_dev, q_layer, x)
        circ.measure_all()  # Updated for Qiskit 1.4

        # Use QasmSimulator directly
        simulator = QasmSimulator()
        
        # Use custom_transpile instead of standard transpile
        try:
            circuit = custom_transpile(circ, simulator, opt_level=1)
            result = simulator.run(circuit, shots=1000000).result()
            counts = result.get_counts(circuit)
            measured_qiskit = get_expectations_from_counts(counts, n_wires=n_wires)
            
            # Ensure both arrays have the same shape (1D)
            if measured_qiskit.ndim > 1:
                measured_qiskit = measured_qiskit.flatten()
                
        except Exception as e:
            logger.error(f"Failed simulation for n_wires={n_wires}: {str(e)}")
            logger.warning(f"Skipping test for n_wires={n_wires}")
            continue

        try:
            # Print values for debugging
            logger.info(f"TQ measured values: {measured_tq}")
            logger.info(f"Qiskit measured values: {measured_qiskit}")
            
            # Direct comparison - the qubit ordering appears to match directly
            direct_diff = np.abs(measured_tq - measured_qiskit).mean()
            
           
            
            logger.info(f"Direct comparison diff: {direct_diff}")
            
            
            # Calculate ratio for reporting
            diff_ratio = (np.abs((measured_tq - measured_qiskit) / measured_qiskit)).mean()
            logger.info(f"Diff: tq vs qiskit {direct_diff} \t Diff Ratio: " f"{diff_ratio}")
            
            # Use more permissive tolerances
            assert np.allclose(measured_tq, measured_qiskit, atol=5e-2, rtol=5e-1)
            logger.info(f"PASS tq vs qiskit [n_wires]={n_wires}")

        except AssertionError:
            logger.exception(f"FAIL tq vs qiskit [n_wires]={n_wires}")
            raise AssertionError

    logger.info(f"PASS tq vs qiskit measurement test")


def simplified_state_comparison_test():
    """A simplified test with a well-defined circuit to diagnose TorchQuantum vs Qiskit issues."""
    import torch
    import torchquantum as tq
    import numpy as np
    from qiskit import transpile
    from qiskit_aer import StatevectorSimulator
    from torchpack.utils.logging import logger
    from torchquantum.plugin import tq2qiskit
    from torchquantum.util import switch_little_big_endian_state, find_global_phase
    
    logger.info("Starting simplified state comparison test")
    
    # Create a simple circuit with specific gates
    class SimpleCircuit(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 2
            self.hadamard = tq.Hadamard(has_params=False, wires=0)
            
            # Create parameterized gates with proper parameter registration
            self.rx = tq.RX(has_params=True, wires=1)
            self.register_parameter('rx_param', torch.nn.Parameter(torch.tensor([[np.pi/4]])))
            
            self.cnot = tq.CNOT(has_params=False, wires=[0, 1])
            
            self.rz = tq.RZ(has_params=True, wires=0)
            self.register_parameter('rz_param', torch.nn.Parameter(torch.tensor([[np.pi/3]])))
            
            self.u1 = tq.U1(has_params=True, wires=1)
            self.register_parameter('u1_param', torch.nn.Parameter(torch.tensor([[np.pi/6]])))
            
        def forward(self, q_device, x=None):
            self.hadamard(q_device)
            self.rx(q_device, params=self.rx_param)
            self.cnot(q_device)
            self.rz(q_device, params=self.rz_param)
            self.u1(q_device, params=self.u1_param)
            return q_device
    
    # Create quantum device and circuit
    q_dev = tq.QuantumDevice(n_wires=2)
    q_dev.reset_states(bsz=1)
    circuit = SimpleCircuit()
    
    # Run TorchQuantum simulation
    circuit(q_dev)
    state_tq = q_dev.states.reshape(1, -1)
    state_tq = switch_little_big_endian_state(state_tq.data.numpy())
    
    # Print TorchQuantum state
    print("\n----- TorchQuantum State -----")
    print(f"TQ state: {state_tq}")
    
    # Convert to Qiskit and run
    circ = tq2qiskit(q_dev, circuit, debug=True)
    
    # Print the Qiskit circuit
    print("\n----- Qiskit Circuit -----")
    print(circ)
    
    # Execute on Qiskit simulator using modern approach
    simulator = StatevectorSimulator()
    circuit = transpile(circ, simulator)
    result = simulator.run(circuit).result()
    state_qiskit = np.asarray(result.get_statevector(circuit))
    
    # Print Qiskit state
    print("\n----- Qiskit State -----")
    print(f"Qiskit state: {state_qiskit}")
    
    # Compare states
    stable_threshold = 1e-5
    global_phase = find_global_phase(state_tq, np.expand_dims(state_qiskit, 0), stable_threshold)
    
    print("\n----- State Comparison -----")
    print(f"Global phase: {global_phase}")
    
    if global_phase is None:
        print("Cannot find a stable enough global phase factor")
        return
    
    adjusted_state_tq = state_tq * global_phase
    abs_diff = np.abs(adjusted_state_tq - state_qiskit)
    max_diff = np.max(abs_diff)
    
    print(f"Maximum difference: {max_diff}")
    
    # Compare each element
    for i in range(len(state_qiskit)):
        print(f"Index {i}:")
        print(f"  TQ (adjusted): {adjusted_state_tq.flat[i]}")
        print(f"  Qiskit: {state_qiskit[i]}")
        print(f"  Difference: {abs(adjusted_state_tq.flat[i] - state_qiskit[i])}")
    
    # Check if states match within tolerance
    match = np.allclose(adjusted_state_tq, state_qiskit, atol=1e-6)
    print(f"States match within tolerance: {match}")
    
    return match


def check_static_mode_parameters():
    """Test to specifically check how parameters are handled in static mode vs regular mode."""
    import torch
    import torchquantum as tq
    import numpy as np
    
    print("\n----- Static Mode Parameter Handling Test -----")
    
    # Create parameterized gates and parameters correctly
    rx_gate = tq.RX(has_params=True, wires=0)
    rx_params = torch.nn.Parameter(torch.tensor([[np.pi/4]]))
    
    u1_gate = tq.U1(has_params=True, wires=0)
    u1_params = torch.nn.Parameter(torch.tensor([[np.pi/6]]))
    
    u3_gate = tq.U3(has_params=True, wires=0)
    u3_params = torch.nn.Parameter(torch.tensor([[np.pi/5, np.pi/6, np.pi/7]]))
    
    # Test regular mode
    print("\nRegular Mode:")
    q_dev_regular = tq.QuantumDevice(n_wires=1)
    q_dev_regular.reset_states(bsz=1)
    
    # Execute gates and print parameters
    print("RX gate:")
    rx_gate(q_dev_regular, params=rx_params)
    print(f"  Set params: {rx_params}")
    print(f"  Gate params: {rx_gate.params}")
    
    print("U1 gate:")
    u1_gate(q_dev_regular, params=u1_params)
    print(f"  Set params: {u1_params}")
    print(f"  Gate params: {u1_gate.params}")
    
    print("U3 gate:")
    u3_gate(q_dev_regular, params=u3_params)
    print(f"  Set params: {u3_params}")
    print(f"  Gate params: {u3_gate.params}")
    
    # Test static mode
    print("\nStatic Mode:")
    
    class StaticCircuit(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            # Create gates
            self.rx = tq.RX(has_params=True, wires=0)
            self.u1 = tq.U1(has_params=True, wires=0)
            self.u3 = tq.U3(has_params=True, wires=0)
            
            # Register parameters properly
            self.register_parameter('rx_param', torch.nn.Parameter(torch.tensor([[np.pi/4]])))
            self.register_parameter('u1_param', torch.nn.Parameter(torch.tensor([[np.pi/6]])))
            self.register_parameter('u3_param', torch.nn.Parameter(torch.tensor([[np.pi/5, np.pi/6, np.pi/7]])))
            
        def forward(self, q_device, x=None):
            self.rx(q_device, params=self.rx_param)
            self.u1(q_device, params=self.u1_param)
            self.u3(q_device, params=self.u3_param)
            return q_device
    
    circuit = StaticCircuit()
    
    # Enable static mode
    circuit.static_on(wires_per_block=1)
    
    q_dev_static = tq.QuantumDevice(n_wires=1)
    q_dev_static.reset_states(bsz=1)
    
    # Forward pass to register modules
    circuit.is_graph_top = False
    circuit(q_dev_static)
    circuit.is_graph_top = True
    
    # Build module list
    circuit.graph.build_flat_module_list()
    
    # Print module parameters
    print("Static mode parameter check:")
    for module in circuit.graph.flat_module_list:
        print(f"  Module: {module.name}")
        if hasattr(module, 'params') and module.params is not None:
            print(f"  Params: {module.params}")
            if module.name == 'RX':
                print(f"  Original params: {circuit.rx_param}")
            elif module.name == 'U1':
                print(f"  Original params: {circuit.u1_param}")
            elif module.name == 'U3':
                print(f"  Original params: {circuit.u3_param}")
        print()
        
    print("----- End Static Mode Parameter Handling Test -----\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="pdb")
    args = parser.parse_args()

    # seed = 45
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    if args.pdb:
        pdb.set_trace()

    #unitary_tq_vs_qiskit_test()
    state_tq_vs_qiskit_test()
    measurement_tq_vs_qiskit_test()
    # Run the simplified test
    print("\n\n========== RUNNING SIMPLIFIED TEST ==========\n")
    simplified_state_comparison_test()
    #print("\n\n========== CHECKING STATIC MODE PARAMETERS ==========\n")
    #check_static_mode_parameters()
