from torchquantum import tq


def random_layer_test():
    q_layer = tq.RandomLayerAllTypes(n_ops=100, wires=[2, 1, 0],
                                     qiskit_comparible=True)
    print(q_layer)

    q_layer = tq.RandomLayer(n_ops=100, wires=[8, 9, 10])
    print(q_layer)


if __name__ == '__main__':
    random_layer_test()
