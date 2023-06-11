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

import random
from copy import deepcopy

import torch
from circ_dag_converter import GATE_DICT, build_my_noise_dict, noise_model_test
from qiskit import QuantumCircuit
from qiskit.compiler import assemble
from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeJakarta, FakeLima
from qiskit.providers.models import BackendProperties

# prop is an input from circ_dag_converter


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
            elif item["name"] == "T2":
                item["value"] = mydict["qubit"][idx]["T2"]
            elif item["name"] == "prob_meas0_prep1":
                item["value"] = mydict["qubit"][idx]["prob_meas0_prep1"]
            elif item["name"] == "prob_meas1_prep0":
                item["value"] = mydict["qubit"][idx]["prob_meas1_prep0"]
            elif item["name"] == "readout_error":
                item["value"] = (
                    mydict["qubit"][idx]["prob_meas1_prep0"]
                    + mydict["qubit"][idx]["prob_meas0_prep1"]
                ) / 2
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
def run_simulation(my_dict, native_circ, mapping_info):
    my_dict = get_randomized_mydict(my_dict)
    backend = get_modified_backend(my_dict)
    simulator = AerSimulator.from_backend(backend)
    shots = 2048
    sim_circ = assemble(
        native_circ, backend=simulator, shots=shots, initial_layout=mapping_info
    )
    result = simulator.run(sim_circ).result()
    counts = result.get_counts()
    fidelity = get_from(counts, "00") / shots
    return native_circ, my_dict, fidelity


def main():
    backend = FakeJakarta()
    noise_model_test(backend)
    mydict = build_my_noise_dict(backend.properties().to_dict())
    new_mydict = get_randomized_mydict(mydict)
    new_backend = get_modified_backend(backend, new_mydict)
    noise_model_test(new_backend)


if __name__ == "__main__":
    main()
