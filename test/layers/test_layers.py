import torchquantum as tq
from qiskit.circuit.library import TwoLocal

for entanglement_type in ("linear", "circular", "full"):
    for n_wires in (3, 5, 10):
        for reps in range(5):

            # create the TQ circuit
            two2 = tq.layer.TwoLocal([tq.RY, tq.RZ], [tq.CZ], arch={"n_wires": n_wires}, entanglement_layer=entanglement_type, reps=reps)
            qdev = tq.QuantumDevice(n_wires, record_op=True)
            two2(qdev)

            # create the qiskit circuit
            two = TwoLocal(n_wires, ['ry','rz'], 'cz', entanglement_type, reps=reps, insert_barriers=False)
            operations = []
            for bit in two.decompose():
                wires = []
                for qu in bit.qubits:
                    wires.append(qu.index)
                operations.append({
                    "name": bit.operation.name,
                    "wires": tuple(wires),
                })

            # create operations list
            qiskit_ops = operations
            tq_ops = ([{"name": op["name"], "wires": (op["wires"],) if isinstance(op["wires"], int) else tuple(op["wires"])} for op in qdev.op_history])

            # create tuples (NOTE: WILL LOSE ORDER SO NOT ENTIRELY CORRECT)
            tq_ops_tuple = {tuple(op) for op in tq_ops}
            qiskit_ops_tuple = {tuple(op) for op in qiskit_ops}

            # assert if they are the same
            test_info = f"{entanglement_type} with {n_wires} wires and {reps} reps"
            assert len(tq_ops) == len(qiskit_ops), f"operations are varying lengths for {test_info}"
            assert tq_ops_tuple == qiskit_ops_tuple, f"operations do not match for {test_info}"

