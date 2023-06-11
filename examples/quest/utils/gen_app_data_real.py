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

import os
import pickle
from qiskit import QuantumCircuit

# from qiskit.providers.fake_provider import *
from .rand_circ_native import *
from qiskit import IBMQ, transpile
import sys

dir_path = "./application_qasm"
files = os.listdir(dir_path)

# ini_backend = FakeGuadalupe()


def IBMQ_ini(backend_str):
    IBMQ.load_account()
    # provider = IBMQ.get_provider(hub="ibm-q-research", group="MIT-1", project="main")
    provider = IBMQ.get_provider(hub="ibm-q-ornl", group="anl", project="csc428")
    backend = provider.get_backend(backend_str)
    return backend


def simu(circ, backend, n_used, shots):
    result = backend.run(circ, shots=shots).result()
    counts = result.get_counts()
    fidelity = counts[n_used * "0"] / shots
    return fidelity


for file_name in files:
    data = []

    circ_app = QuantumCircuit.from_qasm_file(dir_path + "/" + file_name)
    if circ_app.num_qubits > 7:
        # ini_backend = FakeGuadalupe()
        continue
    else:
        # ini_backend = FakeJakarta()
        backend = IBMQ_ini(sys.argv[1])

    for i in range(50):
        props = backend.properties().to_dict()
        my_dict = build_my_noise_dict(props)
        # my_dict = get_randomized_mydict(my_dict)
        # backend = get_modified_backend(ini_backend, my_dict)
        # simulator = AerSimulator.from_backend(backend)

        circ_app.remove_final_measurements()
        transpiled = transpile(circ_app, backend)
        appended = circ_app.compose(circ_app.inverse())
        appended.measure_active()
        appended = transpile(appended, backend)

        try:
            fidelity = simu(appended, backend, appended.num_clbits, shots=4096)
        except Exception as e:
            print("bypassed")

        data_p = [transpiled, my_dict, fidelity]
        data.append(data_p)
        # print(fidelity)
    file_open = open(
        "./app_real_data_test/" + sys.argv[1] + file_name[:-4] + "data", "wb"
    )
    pickle.dump(data, file_open)
    file_open.close()
