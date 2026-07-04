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

import string

import networkx as nx
import torch
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeJakarta, FakeLima
from qiskit.transpiler.passes import RemoveBarriers
from torch_geometric.utils.convert import from_networkx

GATE_DICT = {"rz": 0, "x": 1, "sx": 2, "cx": 3}
NUM_ERROR_DATA = 7
NUM_NODE_TYPE = 2 + len(GATE_DICT)


def get_global_features(circ):
    data = torch.zeros((1, 6))
    data[0][0] = circ.depth()
    data[0][1] = circ.width()
    for key in GATE_DICT:
        if key in circ.count_ops():
            data[0][2 + GATE_DICT[key]] = circ.count_ops()[key]

    return data


def circ_to_dag_with_data(circ, mydict, n_qubit=10):
    # data format: [node_type(onehot)]+[qubit_idx(one or two-hot)]+[T1,T2,T1,T2,gate error,roerror,roerror]+[gate_idx]
    circ = circ.copy()
    circ = RemoveBarriers()(circ)

    dag = circuit_to_dag(circ)
    dag = dag.to_networkx()
    dag_list = list(dag.nodes())
    used_qubit_idx_list = {}
    used_qubit_idx = 0
    for node in dag_list:
        node_type, qubit_idxs, noise_info = data_generator(node, mydict)
        if node_type == "in":
            succnodes = dag.succ[node]
            for succnode in succnodes:
                succnode_type, _, _ = data_generator(succnode, mydict)
                if succnode_type == "out":
                    dag.remove_node(node)
                    dag.remove_node(succnode)
    dag_list = list(dag.nodes())
    for node_idx, node in enumerate(dag_list):
        node_type, qubit_idxs, noise_info = data_generator(node, mydict)
        for qubit_idx in qubit_idxs:
            if not qubit_idx in used_qubit_idx_list:
                used_qubit_idx_list[qubit_idx] = used_qubit_idx
                used_qubit_idx += 1
        data = torch.zeros(NUM_NODE_TYPE + n_qubit + NUM_ERROR_DATA + 1)
        if node_type == "in":
            data[0] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
            data[NUM_NODE_TYPE + n_qubit] = noise_info[0]["T1"]
            data[NUM_NODE_TYPE + n_qubit + 1] = noise_info[0]["T2"]
        elif node_type == "out":
            data[1] = 1
            data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[0]]] = 1
            data[NUM_NODE_TYPE + n_qubit] = noise_info[0]["T1"]
            data[NUM_NODE_TYPE + n_qubit + 1] = noise_info[0]["T2"]
            data[NUM_NODE_TYPE + n_qubit + 5] = noise_info[0]["prob_meas0_prep1"]
            data[NUM_NODE_TYPE + n_qubit + 6] = noise_info[0]["prob_meas1_prep0"]
        else:
            data[2 + GATE_DICT[node_type]] = 1
            for i in range(len(qubit_idxs)):
                data[NUM_NODE_TYPE + used_qubit_idx_list[qubit_idxs[i]]] = 1
                data[NUM_NODE_TYPE + n_qubit + 2 * i] = noise_info[i]["T1"]
                data[NUM_NODE_TYPE + n_qubit + 2 * i + 1] = noise_info[i]["T2"]
            data[NUM_NODE_TYPE + n_qubit + 4] = noise_info[-1]
        data[-1] = node_idx
        if node in dag.nodes():
            dag.nodes[node]["x"] = data
    mapping = dict(zip(dag, string.ascii_lowercase))
    dag = nx.relabel_nodes(dag, mapping)
    global_features = get_global_features(circ)
    liu_features = get_liu_features(dag_list, used_qubit_idx_list, mydict)
    return networkx_torch_convert(dag, global_features, liu_features)


def get_liu_features(dag_list, used_qubit_idx_list, mydict):
    lius_feature = torch.zeros((1, 110))

    for node_idx, node in enumerate(dag_list):
        node_type, qubit_idxs, noise_info = data_generator(node, mydict)
        if node_type == "rz" or node_type == "x" or node_type == "sx":
            lius_feature[0][used_qubit_idx_list[qubit_idxs[0]]] += 1
        elif node_type == "cx":
            lius_feature[0][
                10
                + used_qubit_idx_list[qubit_idxs[0]] * 10
                + used_qubit_idx_list[qubit_idxs[1]]
            ] += 1
    return lius_feature


def networkx_torch_convert(dag, global_features, liu_features):
    myedge = []
    for item in dag.edges:
        myedge.append((item[0], item[1]))
    G = nx.DiGraph()
    G.add_nodes_from(dag._node)
    G.add_edges_from(myedge)
    x = torch.zeros((len(G.nodes()), 24))
    for idx, node in enumerate(G.nodes()):
        x[idx] = dag.nodes[node]["x"]
    G = from_networkx(G)
    G.x = x
    G.global_features = global_features
    G.liu_features = liu_features
    return G


def data_generator(node, mydict):
    if isinstance(node, DAGInNode):
        qubit_idx = int(node.wire._index)
        return "in", [qubit_idx], [mydict["qubit"][qubit_idx]]
    elif isinstance(node, DAGOutNode):
        qubit_idx = int(node.wire._index)
        return "out", [qubit_idx], [mydict["qubit"][qubit_idx]]
    elif isinstance(node, DAGOpNode):
        name = node.name
        qargs = node.qargs
        qubit_list = []
        for qubit in qargs:
            qubit_list.append(qubit._index)
        mylist = [mydict["qubit"][qubit_idx] for qubit_idx in qubit_list]
        mylist.append(mydict["gate"][tuple(qubit_list)][name])
        return (name, qubit_list, mylist)
    else:
        raise NotImplementedError("Unknown node type")


def build_my_noise_dict(prop):
    mydict = {}
    mydict["qubit"] = {}
    mydict["gate"] = {}
    for i, qubit_prop in enumerate(prop["qubits"]):
        mydict["qubit"][i] = {}
        for item in qubit_prop:
            if item["name"] == "T1":
                mydict["qubit"][i]["T1"] = item["value"]
            elif item["name"] == "T2":
                mydict["qubit"][i]["T2"] = item["value"]
            elif item["name"] == "prob_meas0_prep1":
                mydict["qubit"][i]["prob_meas0_prep1"] = item["value"]
            elif item["name"] == "prob_meas1_prep0":
                mydict["qubit"][i]["prob_meas1_prep0"] = item["value"]
    for gate_prop in prop["gates"]:
        if not gate_prop["gate"] in GATE_DICT:
            continue
        qubit_list = tuple(gate_prop["qubits"])
        if qubit_list not in mydict["gate"]:
            mydict["gate"][qubit_list] = {}
        for item in gate_prop["parameters"]:
            if item["name"] == "gate_error":
                mydict["gate"][qubit_list][gate_prop["gate"]] = item["value"]
    return mydict


# def noise_model_test(backend):
#     # test which parameters are useful in determining the noise model
#     circ = QuantumCircuit(2)
#     circ.h(0)
#     circ.cnot(0, 1)
#     circ.save_density_matrix()
#     simulator = AerSimulator.from_backend(backend)
#     simulator.run(circ)
#     result = simulator.run(circ).result()
#     noise_dm = result.data()["density_matrix"].data
#     print(noise_dm)


def main():
    backend = FakeJakarta()
    props = backend.properties().to_dict()
    mydict = build_my_noise_dict(props)
    circ = QuantumCircuit(3)
    circ.cnot(1, 0)
    circ = transpile(circ, backend)
    # print(circ_global_features(circ))
    # print(mydict)
    dag = circ_to_dag_with_data(circ, mydict)
    dag.y = 1
    print(dag.global_features)


if __name__ == "__main__":
    main()
