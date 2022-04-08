import torch
import argparse

import torchquantum as tq
import torchquantum.functional as tqf

import random
import numpy as np


class QLayer(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50,
                                           wires=list(range(self.n_wires)))

        # gates with trainable parameters
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

        self.measure = tq.MeasureMultiPauliSum(
            obs_list=[{'wires': [0, 2, 3, 1],
                       'observables': ['x', 'y', 'z', 'i'],
                       'coefficient': [1, 0.5, 0.4, 0.3]
                       },
                      {'wires': [0, 2, 3, 1],
                       'observables': ['x', 'x', 'z', 'i'],
                       'coefficient': [1, 0.5, 0.4, 0.3]
                       }])

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice):
        """
        1. To convert tq QuantumModule to qiskit or run in the static
        model, need to:
            (1) add @tq.static_support before the forward
            (2) make sure to add
                static=self.static_mode and
                parent_graph=self.graph
                to all the tqf functions, such as tqf.hadamard below
        """
        self.q_device = q_device

        self.random_layer(self.q_device)

        # some trainable gates (instantiated ahead of time)
        self.rx0(self.q_device, wires=0)
        self.ry0(self.q_device, wires=1)
        self.rz0(self.q_device, wires=3)
        self.crx0(self.q_device, wires=[0, 2])

        # add some more non-parameterized gates (add on-the-fly)
        tqf.hadamard(self.q_device, wires=3, static=self.static_mode,
                     parent_graph=self.graph)
        tqf.sx(self.q_device, wires=2, static=self.static_mode,
               parent_graph=self.graph)
        tqf.cnot(self.q_device, wires=[3, 0], static=self.static_mode,
                 parent_graph=self.graph)

        return self.measure(self.q_device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', action='store_true', help='debug with pdb')

    args = parser.parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    q_model = QLayer()

    q_device = tq.QuantumDevice(n_wires=4)
    q_device.reset_states(bsz=1)
    res = q_model(q_device)
    print(res)



if __name__ == '__main__':
    main()
