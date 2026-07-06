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

import pickle
import random
import sys

import torch
from qiskit import QuantumCircuit
from torchpack.utils.config import configs
from utils.circ_dag_converter import circ_to_dag_with_data


def load_data_from_raw(file_name):
    file = open("data/raw_data_qasm/" + file_name, "rb")
    data = pickle.load(file)
    file.close()
    print("Size of the data: ", len(data))
    return raw_pyg_converter(data)


def load_data_from_pyg(file_name):
    try:
        return load_normalized_data(file_name)
    except Exception as e:
        try:
            file = open("data/pyg_data/" + file_name, "rb")
            normalize_data(file_name)
        except Exception as e:
            load_data_and_save(file_name)
            normalize_data(file_name)
        return load_normalized_data(file_name)


def load_normalized_data(file_name):
    file = open("data/normalized_data/" + file_name, "rb")
    data = pickle.load(file)
    file.close()
    print("Size of the data: ", len(data))
    return data


def normalize_data(file_name):
    file = open("data/pyg_data/" + file_name, "rb")
    data = pickle.load(file)
    file.close()
    if configs.evalmode:
        file = open("data/normalized_data/" + configs.dataset.name + "meta", "rb")
        meta = pickle.load(file)
        file.close()
        print(meta)
        print(meta[1])
        for k, dag in enumerate(data):
            data[k].x = (dag.x - meta[0]) / (1e-8 + meta[1])
            data[k].global_features = (dag.global_features - meta[2]) / (1e-8 + meta[3])
    else:
        all_features = None
        for k, dag in enumerate(data):
            if not k:
                all_features = dag.x
                global_features = dag.global_features
                liu_features = dag.liu_features
            else:
                all_features = torch.cat([all_features, dag.x])
                global_features = torch.cat([global_features, dag.global_features])
                liu_features = torch.cat([liu_features, dag.liu_features])

        means = all_features.mean(0)
        stds = all_features.std(0)
        means_gf = global_features.mean(0)
        stds_gf = global_features.std(0)
        means_liu = liu_features.mean(0)
        stds_liu = liu_features.std(0)
        for k, dag in enumerate(data):
            data[k].x = (dag.x - means) / (1e-8 + stds)
            data[k].global_features = (dag.global_features - means_gf) / (
                1e-8 + stds_gf
            )
            data[k].liu_features = (dag.liu_features - means_liu) / (1e-8 + stds_liu)
        file = open("data/normalized_data/" + file_name + "meta", "wb")
        pickle.dump([means, stds, means_gf, stds_gf], file)
        file.close()
    file = open("data/normalized_data/" + file_name, "wb")
    pickle.dump(data, file)
    file.close()


def load_data_and_save(file_name):
    file = open("data/raw_data_qasm/" + file_name, "rb")
    data = pickle.load(file)
    file.close()
    pyg_data = raw_pyg_converter(data)
    random.shuffle(pyg_data)
    file = open("data/pyg_data/" + file_name, "wb")
    pickle.dump(pyg_data, file)
    file.close()


def raw_pyg_converter(dataset):
    pygdataset = []
    for data in dataset:
        circ = QuantumCircuit()
        circ = circ.from_qasm_str(data[0])
        dag = circ_to_dag_with_data(circ, data[1])
        dag.y = data[2]
        pygdataset.append(dag)
    return pygdataset


if __name__ == "__main__":
    file_name = sys.argv[1]
    dataset = load_data_and_save(file_name)
