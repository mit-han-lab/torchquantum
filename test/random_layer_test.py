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

from torchquantum import tq


def random_layer_test():
    import pdb

    pdb.set_trace()
    q_layer = tq.RandomLayerAllTypes(n_ops=100, wires=[2, 1, 0], qiskit_compatible=True)
    print(q_layer)

    # q_layer = tq.RandomLayer(n_ops=100, wires=[8, 9, 10])
    # print(q_layer)

    q_layer2 = tq.RandomLayer(n_ops=200, wires=[1, 2, 3])

    q_layer2.rebuild_random_layer_from_op_list(
        n_ops_in=q_layer.n_ops, wires_in=q_layer.wires, op_list_in=q_layer.op_list
    )
    print(q_layer2)


if __name__ == "__main__":
    random_layer_test()
