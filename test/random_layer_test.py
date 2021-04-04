from torchquantum import tq


def random_layer_test():
    import pdb
    pdb.set_trace()
    q_layer = tq.RandomLayerAllTypes(n_ops=100, wires=[2, 1, 0],
                                     qiskit_compatible=True)
    print(q_layer)

    # q_layer = tq.RandomLayer(n_ops=100, wires=[8, 9, 10])
    # print(q_layer)

    q_layer2 = tq.RandomLayer(n_ops=200, wires=[1, 2, 3])

    q_layer2.rebuild_random_layer_from_op_list(n_ops_in=q_layer.n_ops,
                                               wires_in=q_layer.wires,
                                               op_list_in=q_layer.op_list)
    print(q_layer2)


if __name__ == '__main__':
    random_layer_test()
