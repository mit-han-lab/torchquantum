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

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional.functionals as tqf
import numpy as np

from enum import IntEnum
from torchquantum.functional import mat_dict
from torchquantum.util.quantization.clifford_quantization import CliffordQuantizer
from abc import ABCMeta
from ..macro import C_DTYPE, F_DTYPE
from torchpack.utils.logging import logger
from typing import Iterable, Union, List

from .op_types import *
from .hadamard import *
from .paulix import *
from .pauliy import *
from .pauliz import *
from .i import *
from .s import *
from .t import *
from .sx import *
from .swap import *
from .toffoli import *
from .rx import *
from .ry import *
from .rz import *
from .r import *
from .iswap import *
from .ecr import *
from .single_excitation import *
from .global_phase import *
from .phase_shift import *
from .rot import *
from .trainable_unitary import *
from .qft import *
from .xx_min_yy import *
from .xx_plus_yy import *
from .reset import *
from .qubit_unitary import *
from .u1 import *
from .u2 import *
from .u3 import *

__all__ = [
    "op_name_dict",
    "Operator",
    "Operation",
    "DiagonalOperation",
    "Observable",
    "Hadamard",
    "H",
    "SHadamard",
    "SH",
    "PauliX",
    "PauliY",
    "PauliZ",
    "I",
    "S",
    "T",
    "SX",
    "CNOT",
    "CZ",
    "CY",
    "RX",
    "RY",
    "RZ",
    "RXX",
    "RYY",
    "RZZ",
    "RZX",
    "SWAP",
    "SSWAP",
    "CSWAP",
    "Toffoli",
    "PhaseShift",
    "Rot",
    "MultiRZ",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "U",
    "U1",
    "U2",
    "U3",
    "CU",
    "CU1",
    "CU2",
    "CU3",
    "QubitUnitary",
    "QubitUnitaryFast",
    "TrainableUnitary",
    "TrainableUnitaryStrict",
    "MultiCNOT",
    "MultiXCNOT",
    "Reset",
    "SingleExcitation",
    "EchoedCrossResonance",
    "ECR",
    "QFT",
    "SDG",
    "TDG",
    "SXDG",
    "CH",
    "CCZ",
    "ISWAP",
    "CS",
    "CSDG",
    "CSX",
    "CHadamard",
    "CCZ",
    "DCX",
    "XXMINYY",
    "XXPLUSYY",
    "C3X",
    "R",
    "C4X",
    "RC3X",
    "RCCX",
    "GlobalPhase",
    "C3SX",
]


op_name_dict = {
    "hadamard": Hadamard,
    "h": Hadamard,
    "shadamard": SHadamard,
    "sh": SHadamard,
    "paulix": PauliX,
    "x": PauliX,
    "pauliy": PauliY,
    "y": PauliY,
    "pauliz": PauliZ,
    "z": PauliZ,
    "i": I,
    "s": S,
    "t": T,
    "sx": SX,
    "cx": CNOT,
    "cnot": CNOT,
    "cz": CZ,
    "cy": CY,
    "rx": RX,
    "ry": RY,
    "rz": RZ,
    "rxx": RXX,
    "xx": RXX,
    "ryy": RYY,
    "yy": RYY,
    "rzz": RZZ,
    "zz": RZZ,
    "rzx": RZX,
    "zx": RZX,
    "swap": SWAP,
    "sswap": SSWAP,
    "cswap": CSWAP,
    "toffoli": Toffoli,
    "ccx": Toffoli,
    "phaseshift": PhaseShift,
    "rot": Rot,
    "multirz": MultiRZ,
    "crx": CRX,
    "cry": CRY,
    "crz": CRZ,
    "crot": CRot,
    "u1": U1,
    "p": U1,
    "u2": U2,
    "u3": U3,
    "u": U3,
    "cu1": CU1,
    "cp": CU1,
    "cr": CU1,
    "cphase": CU1,
    "cu2": CU2,
    "cu3": CU3,
    "cu": CU,
    "qubitunitary": QubitUnitary,
    "qubitunitarystrict": QubitUnitaryFast,
    "qubitunitaryfast": QubitUnitaryFast,
    "trainableunitary": TrainableUnitary,
    "trainableunitarystrict": TrainableUnitaryStrict,
    "multicnot": MultiCNOT,
    "multixcnot": MultiXCNOT,
    "reset": Reset,
    "singleexcitation": SingleExcitation,
    "ecr": ECR,
    "echoedcrossresonance": ECR,
    "QFT": QFT,
    "sdg": SDG,
    "cs": CS,
    "chadamard": CHadamard,
    "ch": CH,
    "dcx": DCX,
    "xxminyy": XXMINYY,
    "xxplusyy": XXPLUSYY,
    "c3x": C3X,
    "tdg": TDG,
    "sxdg": SXDG,
    "ch": CH,
    "ccz": CCZ,
    "iswap": ISWAP,
    "csdg": CSDG,
    "csx": CSX,
    "r": R,
    "c3sx": C3SX,
    "globalphase": GlobalPhase,
    "rccx": RCCX,
    "rc3x": RC3X,
    "c4x": C4X,
}
