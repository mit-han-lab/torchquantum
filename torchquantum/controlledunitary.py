import functools
import torch
import numpy as np

import torchquantum as tq

from typing import Callable, Union, Optional, List, Dict, TYPE_CHECKING
from .macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from .utils import pauli_eigs, diag
from torchpack.utils.logging import logger
from torchquantum.utils import normalize_statevector

if TYPE_CHECKING:
    from torchquantum.devices import QuantumDevice
else:
    QuantumDevice = None
    
    
import copy


