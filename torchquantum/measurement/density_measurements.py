import random

import torch
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from torchquantum.macro import F_DTYPE

from typing import Union, List
from collections import Counter, OrderedDict

from torchquantum.functional import mat_dict
# from .operator import op_name_dict, Observable
import torchquantum.operator as op
from copy import deepcopy
import matplotlib.pyplot as plt

__all__ = [
    "expval_joint_sampling_grouping",
    "expval_joint_analytical",
    "expval_joint_sampling",
    "expval",
    "measure",
]


def measure(noisedev: tq.NoiseDevice, n_shots=1024, draw_id=None):
    return



def expval_joint_sampling_grouping(
        qdev: tq.NoiseDevice,
        observables: List[str],
        n_shots_per_group=1024,
):
    return


def expval_joint_sampling(
        qdev: tq.NoiseDevice,
        observable: str,
        n_shots=1024,
):
    return


def expval_joint_analytical(
        qdev: tq.NoiseDevice,
        observable: str,
):
    return


def expval(
        qdev: tq.NoiseDevice,
        wires: Union[int, List[int]],
        observables: Union[op.Observable, List[op.Observable]],
):
    return







if __name__ == '__main__':
    print("")
