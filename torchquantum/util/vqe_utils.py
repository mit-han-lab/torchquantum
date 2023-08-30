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

import itertools
import numpy as np


def parse_hamiltonian_file(filename: str) -> dict:
    """
       Parses a Hamiltonian file and returns the Hamiltonian information as a dictionary.

       Args:
           filename (str): The name of the Hamiltonian file.

       Returns:
           dict: A dictionary containing the Hamiltonian information.

       """
    hamil_info = {}
    with open(filename, "r") as rfid:
        lines = rfid.read().split("\n")
        hamil_list = []
        for k, line in enumerate(lines):
            if not line.strip():
                continue
            line = line.strip()
            hamil = {"wires": [], "observables": []}
            if k == 0:
                name, method, n_wires = line.split(" ")
                hamil_info["name"] = name
                hamil_info["method"] = method
                hamil_info["n_wires"] = eval(n_wires)
            else:
                info = line.split(" ")
                hamil["coefficient"] = eval(info[0])
                for observable in info[1:]:
                    assert observable[0].lower() in ["x", "y", "z", "i"]
                    hamil["wires"].append(int(eval(observable[1:])))
                    hamil["observables"].append(observable[0].lower())
                hamil_list.append(hamil)

    hamil_info["hamil_list"] = hamil_list

    return hamil_info


def generate_n_hamiltonian(n_wires: int, n_hamil: int, n_lines: int) -> dict:
    """
        Generates a random Hamiltonian with a specified number of wires and terms.

        Args:
            n_wires (int): The number of wires.
            n_hamil (int): The number of terms in the Hamiltonian.
            n_lines (int): The desired number of unique Hamiltonians to generate.

        Returns:
            dict: A dictionary containing the generated Hamiltonian information.

        Raises:
            AssertionError: If n_hamil is not within the range (0, n_wires].

        """
    assert 0 < n_hamil <= n_wires

    hamil_info = {
        "name": "generated",
        "method": f"{n_hamil}_hamil",
        "n_wires": n_wires,
        "hamil_list": [],
    }

    combs = list(map(list, itertools.combinations(range(n_wires), n_hamil)))

    ctr_lines = 0
    while ctr_lines < n_lines:
        hamil = {}
        comb = combs[np.random.choice(len(combs))]
        comb.sort()
        hamil["wires"] = comb
        hamil["observables"] = list(np.random.choice(["x", "y", "z"], n_hamil))
        hamil["coefficient"] = 1.0

        if hamil in hamil_info["hamil_list"]:
            print(hamil)
            continue
        else:
            hamil_info["hamil_list"].append(hamil)
            ctr_lines += 1

    return hamil_info


def test_parse_hamiltonian_file():
    """
        Test function for parse_hamiltonian_file.
        """
    file = "../examples/vqe/h2.txt"
    print(parse_hamiltonian_file(file))


def test_generate_n_hamiltonian():
    """
    Test function for generate_n_hamiltonian.
    """
    print(generate_n_hamiltonian(n_wires=5, n_hamil=3, n_lines=100))


if __name__ == "__main__":

    test_parse_hamiltonian_file()
    test_generate_n_hamiltonian()
