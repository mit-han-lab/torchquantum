import numpy as np
import pickle
import sys
import copy
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from rand_circ_native import *


def noisy_sim(my_dict, circ, backend):
    backend = get_modified_backend(backend, my_dict)
    simulator = AerSimulator.from_backend(backend)
    circ.save_density_matrix()
    result = simulator.run(circ).result()
    noise_dm = result.data()["density_matrix"].data
    return noise_dm.astype(np.complex64)


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


def free_sim(circ):
    backend = Aer.get_backend("aer_simulator")
    circ.save_density_matrix()
    result = backend.run(circ).result()
    noise_dm = result.data()["density_matrix"].data
    return noise_dm.astype(np.complex64)


def load_pick(file_name):
    file_open = open(file_name, "rb")
    data = pickle.load(file_open)
    file_open.close()
    return data


def main():
    file_name = sys.argv[1]
    data = load_pick(file_name)
    fides = []
    psts = []

    backend = FakeManila()
    for dp in data:
        pst = dp[2]
        a = free_sim(copy.deepcopy(dp[0]))
        b = noisy_sim(dp[1], copy.deepcopy(dp[0]), backend=backend)
        fid = np.trace(np.matmul(a, b)).real
        fides.append(fid)
        psts.append(pst)

    print(len(fides))
    print(len(psts))
    file_o = open(sys.argv[1][:-4] + "fidpst", "wb")
    pickle.dump([fides, psts], file_o)
    file_o.close()

    return fides, psts


if __name__ == "__main__":
    main()
