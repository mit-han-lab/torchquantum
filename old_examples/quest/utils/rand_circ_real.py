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

from ast import dump
import pdb
import pickle
import random
import sys
from copy import deepcopy
from turtle import back

import numpy as np
import qiskit
import torch
from qiskit import QuantumCircuit, transpile, IBMQ
from circ_dag_converter import GATE_DICT, build_my_noise_dict
from qiskit.compiler import assemble
from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import *
from qiskit.providers.models import BackendProperties

backend = FakeGuadalupe()
basis_gates = ["rz", "sx", "x", "cx"]


def get_bak_info(backend):
    config = backend.configuration()
    return config.n_qubits, config.basis_gates, config.coupling_map


def avai_coup_map(coupling_map, n_used):
    avai_coup = []
    for pair in coupling_map:
        if (pair[0] < n_used) & (pair[1] < n_used):
            avai_coup.append(pair)
    return [*range(n_used)], avai_coup


def rand_circs(backend, n_circ, n_used, n_gate):
    n_qubits, basis_gates, coupling_map = get_bak_info(backend)
    basis_gates = ["rz", "sx", "x", "cx"]
    avai_qubit, avai_coup = avai_coup_map(coupling_map, n_used)
    assert n_used <= n_qubits
    results = []
    for circ in range(n_circ):
        result = QuantumCircuit(n_used, n_used)
        gates = random.choices(basis_gates, k=n_gate)
        for gate in gates:
            rand_qbit = random.choice(avai_qubit)
            if gate == "cx":
                if n_used > 1:
                    rand_coup = random.choice(avai_coup)
                    result.cx(rand_coup[0], rand_coup[1])
            elif gate == "rz":
                rand_angl = 2 * np.pi * random.random()
                result.rz(rand_angl, rand_qbit)
            elif gate == "sx":
                result.sx(rand_qbit)
            elif gate == "x":
                result.x(rand_qbit)
        transpiled = transpile(result, backend, initial_layout=[*range(n_used)])
        transpiled.barrier()
        results.append(transpiled)
    return results


def append_reverse(circuits, n_used, backend):
    results = []
    for circ in circuits:
        appended = circ.compose(circ.inverse())
        appended = transpile(appended, backend)
        appended.measure([*range(n_used)], [*range(n_used)])
        results.append(appended)
    return results


def get_randomized_mydict(my_dict):
    for i in my_dict:
        for j in my_dict[i]:
            for k in my_dict[i][j]:
                my_dict[i][j][k] *= random.uniform(0.5, 1.5)
    return my_dict


def get_modified_backend(backend, mydict):
    backend = deepcopy(backend)
    prop_dict = backend.properties().to_dict()

    for idx, qubit in enumerate(prop_dict["qubits"]):
        for _, item in enumerate(qubit):
            if item["name"] == "T1":
                item["value"] = mydict["qubit"][idx]["T1"]
            if item["name"] == "T2":
                item["value"] = mydict["qubit"][idx]["T2"]
    for _, gate in enumerate(prop_dict["gates"]):
        if gate["gate"] not in GATE_DICT:
            continue
        for _, parameter in enumerate(gate["parameters"]):
            if parameter["name"] == "gate_error":
                parameter["value"] = mydict["gate"][tuple(gate["qubits"])][gate["gate"]]
    new_prop = BackendProperties.from_dict(prop_dict)
    backend._properties = new_prop
    return backend


def get_from(d: dict, key: str):

    value = 0
    if key in d:
        value = d[key]
    return value


# need total_circ as input from Jinglei
def run_simulation(my_dict, native_circ, backend, n_used):
    my_dict = get_randomized_mydict(my_dict)
    backend = get_modified_backend(backend, my_dict)
    simulator = AerSimulator.from_backend(backend)
    shots = 2048
    sim_circ = assemble(
        native_circ,
        backend=simulator,
        shots=shots,
    )
    result = simulator.run(sim_circ).result()
    counts = result.get_counts()
    fidelity = counts
    fidelity = [count[n_used * "0"] / shots for count in counts]
    return native_circ, my_dict, fidelity


def run_real(my_dict, native_circ, backend, n_used):
    # my_dict = get_randomized_mydict(my_dict)
    # backend = get_modified_backend(backend, my_dict)
    # simulator = AerSimulator.from_backend(backend)
    shots = 2048
    # sim_circ = assemble(
    #     native_circ,
    #     backend=backend,
    #     shots=shots,
    # )
    result = backend.run(native_circ, shots=shots).result()
    counts = result.get_counts()
    fidelity = counts
    fidelity = [count[n_used * "0"] / shots for count in counts]
    return native_circ, my_dict, fidelity


def test():
    n_used = 4
    n_circ = 100
    n_gate = 100
    circs = rand_circs(backend, n_circ, n_used, n_gate)
    appended = append_reverse(circs)


def IBMQ_ini(backend_str):
    IBMQ.load_account()
    # provider = IBMQ.get_provider(hub="ibm-q-research", group="MIT-1", project="main")
    provider = IBMQ.get_provider(hub="ibm-q-ornl", group="anl", project="csc428")
    backend = provider.get_backend(backend_str)
    return backend


def dump_dt(file_name, data, index):
    file = open(file_name + index, "wb")
    pickle.dump(data, file)
    file.close()
    return 0


def main():

    # Number of qubit 1 - 10
    # Number of gates < 500
    # Number of data points 2000
    # 500 circuits, 500 noise model

    # backend = FakeGuadalupe()
    backend = IBMQ_ini(sys.argv[2])

    file_name = sys.argv[1]
    data = []

    n_circ = 100
    for n_used in range(1, 10, 1):
        for n_gate in [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 500]:
            # for model in range(5):
            native_circs = rand_circs(backend, n_circ, n_used, n_gate)
            appended = append_reverse(native_circs, n_used, backend)
            props = backend.properties().to_dict()
            mydict = build_my_noise_dict(props)
            appended_circ, my_dict, fidelity = run_real(
                mydict, appended, backend, n_used
            )
            data_seg = [
                [circ, my_dict, fide] for (circ, fide) in zip(native_circs, fidelity)
            ]
            data = data + data_seg
            dump_dt(file_name, data, str(n_used) + str(n_gate))
    dump_dt(file_name, data, "final" + sys.argv[2])


if __name__ == "__main__":
    main()
