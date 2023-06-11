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
import torchquantum.functional as tqf
import numpy as np

from typing import Any, Iterable, List
from torchquantum.util import pauli_string_to_matrix

__all__ = ["Hamiltonian"]


def parse_hamiltonian_file(filename: str) -> dict:
    """Parse the Hamiltonian file.
    Args:
        filename: The filename of the Hamiltonian file.
    Returns:
        A dictionary containing the information of the Hamiltonian.
    Example file:
        h2 bk 2
        -1.052373245772859 I0 I1
        0.39793742484318045 I0 Z1
        -0.39793742484318045 Z0 I1
        -0.01128010425623538 Z0 Z1
        0.18093119978423156 X0 X1   
    """

    hamil_info = {}
    hamil_info["paulis"] = []
    hamil_info["coeffs"] = []

    with open(filename, "r") as rfid:
        lines = rfid.read().split("\n")

        for k, line in enumerate(lines):
            if not line.strip():
                continue
            line = line.strip()
            if k == 0:
                name, method, n_wires = line.split(" ")
                hamil_info["name"] = name
                hamil_info["method"] = method
                hamil_info["n_wires"] = eval(n_wires)
            else:
                info = line.split(" ")
                hamil_info["coeffs"].append(eval(info[0]))

                pauli_string = ""

                wires = []
                obs = []

                for observable in info[1:]:
                    assert observable[0].upper() in ["X", "Y", "Z", "I"]    
                    wires.append(int(eval(observable[1:])))
                    obs.append(observable[0].upper())

                for i in range(hamil_info["n_wires"]):
                    if i in wires:
                        wire_idx = wires.index(i)
                        pauli_string += obs[wire_idx]
                    else:
                        pauli_string += "I"

                hamil_info["paulis"].append(pauli_string)
    
    return hamil_info

class Hamiltonian(object):
    """Hamiltonian class."""
    def __init__(self,
                 coeffs: List[float],
                 paulis: List[str],
                 endianness: str = "big",
                 ) -> None:
        """Initialize the Hamiltonian.
        Args:
            coeffs: The coefficients of the Hamiltonian.
            paulis: The operators of the Hamiltonian, described in strings.
            endianness: The endianness of the operators. Default is big. Qubit 0 is the most significant bit.
        Example:

        .. code-block:: python
            coeffs = [1.0, 1.0]
            paulis = ["ZZ", "ZX"]
            hamil = tq.Hamiltonian(coeffs, paulis)

        """
        if endianness not in ["big", "little"]:
            raise ValueError("Endianness must be either big or little.")
        if len(coeffs) != len(paulis):
            raise ValueError("The number of coefficients and operators must be the same.")
        for op in paulis:
            if len(op) != len(paulis[0]):
                raise ValueError("The length of each operator must be the same.")
            for char in op:
                if char not in ["X", "Y", "Z", "I"]:
                    raise ValueError("The operator must be a string of X, Y, Z, and I.")
        
        self.n_wires = len(paulis[0])
        self.coeffs = coeffs
        self.paulis = paulis
        self.endianness = endianness
        if self.endianness == "little":
            self.paulis = [pauli[::-1] for pauli in self.paulis]
    
    @property
    def matrix(self) -> torch.Tensor:
        """Return the matrix of the Hamiltonian."""
        return self.get_matrix()
    
    def get_matrix(self) -> torch.Tensor:
        """Return the matrix of the Hamiltonian."""
        matrix = self.coeffs[0] * pauli_string_to_matrix(self.paulis[0])
        for coeff, pauli in zip(self.coeffs[1:], self.paulis[1:]):
            matrix += coeff * pauli_string_to_matrix(pauli)

        return matrix
    
    def __repr__(self) -> str:
        """Return the representation string."""
        return f"{self.__class__.__name__}({self.coeffs}, {self.paulis}, {self.endianness})"
    
    def __len__(self) -> int:
        """Return the number of terms in the Hamiltonian."""
        return len(self.coeffs)
    
    @classmethod
    def from_file(cls, file_path: str) -> Any:
        """Initialize the Hamiltonian from a file.
        Args:
            file_path: The path to the file.
        Example:

        .. code-block:: python
            hamil = tq.Hamiltonian.from_file("hamiltonian.txt")

        Example of the hamiltonian.txt file:
        h2 bk 2
        -1.052373245772859 I0 I1
        0.39793742484318045 I0 Z1
        -0.39793742484318045 Z0 I1
        -0.01128010425623538 Z0 Z1
        0.18093119978423156 X0 X1            

        """
        hamil_info = parse_hamiltonian_file(file_path)
        return cls(hamil_info["coeffs"], hamil_info["paulis"])


if __name__ == '__main__':
    import pdb
    pdb.set_trace()

    coeffs = [1.0, 1.0]
    paulis = ["ZZ", "ZX"]
    hamil = Hamiltonian(coeffs, paulis)
    print(hamil.get_matrix())

    coeffs = [0.6]
    paulis = ["XXZ"]
    hamil = Hamiltonian(coeffs, paulis)
    print(hamil.get_matrix())

    coeffs = [1.0]
    paulis = ["II"]
    hamil = Hamiltonian(coeffs, paulis)
    print(hamil.get_matrix())

    hamil = Hamiltonian.from_file("torchquantum/algorithms/h2.txt")
    print(hamil.matrix)

    

