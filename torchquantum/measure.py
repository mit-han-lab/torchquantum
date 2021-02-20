import functools
import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
from typing import Union, List


def expval(q_device: tq.QuantumDevice,
           wires: Union[int, List[int]],
           observables: Union[tq.Observable, List[tq.Observable]]):

    all_dims = np.arange(q_device.states.dim())
    if isinstance(wires, int):
        wires = [wires]
        observables = [observables]

    # rotation to the desired basis
    for wire, observable in zip(wires, observables):
        for rotation in observable.diagonalizing_gates():
            rotation(q_device, wires=wire)

    states = q_device.states
    # compute magnitude
    state_mag = torch.abs(states) ** 2

    expectations = []
    for wire, observable in zip(wires, observables):
        # compute marginal magnitude
        reduction_dims = np.delete(all_dims, [0, wire + 1])
        probs = state_mag.sum(list(reduction_dims))
        res = probs.matmul(observable.eigvals.to(probs))
        expectations.append(res)

    return torch.stack(expectations, dim=-1)

