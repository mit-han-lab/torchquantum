import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

from typing import Iterable
from torchquantum.plugins.qiskit_macros import QISKIT_INCOMPATIBLE_FUNC_NAMES
from torchpack.utils.logging import logger
from torchquantum.vqe_utils import parse_hamiltonian_file


__all__ = ["Hamiltonian"]

class Hamiltonian(object):
    def __init__(self, hamil_info) -> None:
        self.hamil_info = self.process_hamil_info(hamil_info)
        self.n_wires = hamil_info["n_wires"]

    def process_hamil_info(self, hamil_info):
        hamil_list = hamil_info["hamil_list"]
        n_wires = hamil_info["n_wires"]
        all_info = []

        for hamil in hamil_list:
            pauli_string = ""
            for i in range(n_wires):
                if i in hamil["wires"]:
                    wire = hamil["wires"].index(i)
                    pauli_string += hamil["observables"][wire].upper()
                else:
                    pauli_string += "I"
            all_info.append({"pauli_string": pauli_string, "coeff": hamil["coefficient"]})
        hamil_info["hamil_list"] = all_info
        return hamil_info

    @classmethod
    def from_file(cls, file_path):
        hamil_info = parse_hamiltonian_file(file_path)
        return cls(hamil_info)
    