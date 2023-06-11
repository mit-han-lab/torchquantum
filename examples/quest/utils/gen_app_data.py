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
from qiskit.providers.fake_provider import *
from .rand_circ_native import *
from copy import deepcopy

dir_path = "../application_qasm"
files = os.listdir(dir_path)

# ini_backend = FakeGuadalupe()


def simu(circ, backend, n_used, shots):
    result = backend.run(circ, shots=shots).result()
    counts = result.get_counts()
    fidelity = 0
    if n_used * "0" in counts:
        fidelity = counts[n_used * "0"] / shots
    return fidelity


def noise_level(my_dict, level):
    for i in my_dict:
        for j in my_dict[i]:
            for k in my_dict[i][j]:
                if (k != "T1") & (k != "T2"):
                    my_dict[i][j][k] *= level
    return my_dict


noise_levels = np.exp2(np.linspace(-3, 6, num=350))

all_data = []
for file_name in files:
    data = []
    try:
        circ_app = QuantumCircuit.from_qasm_file(dir_path + "/" + file_name)
    except Exception as e:
        print("CANNOT LOAD")
        continue

    if circ_app.num_qubits > 7:
        ini_backend = FakeGuadalupe()
        continue
    else:
        ini_backend = FakeJakarta()

    circ_app.remove_final_measurements()
    transpiled = transpile(circ_app, ini_backend)
    try:
        appended = circ_app.compose(circ_app.inverse())
    except Exception as e:
        print("ERROR!")
        continue
    appended.measure_active()
    appended = transpile(appended, ini_backend)

    for level in noise_levels:
        props = ini_backend.properties().to_dict()
        my_dict = build_my_noise_dict(props)
        my_dict = get_randomized_mydict(my_dict)
        my_dict = noise_level(my_dict, level)
        backend = get_modified_backend(deepcopy(ini_backend), my_dict)
        simulator = AerSimulator.from_backend(backend)

        try:
            fidelity = simu(appended, simulator, appended.num_clbits, shots=1024)
        except Exception as e:
            print("bypassed")
            continue

        data_p = [transpiled, my_dict, fidelity]
        data.append(data_p)
    file_open = open("../app_data_test/" + file_name[:-4] + "data", "wb")
    pickle.dump(data, file_open)
    file_open.close()
    all_data = all_data + deepcopy(data)
    file_open = open("../app_data_test/ALL_APP_DATA.data", "wb")
    pickle.dump(all_data, file_open)
    file_open.close()
