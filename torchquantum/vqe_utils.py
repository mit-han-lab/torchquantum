import itertools
import numpy as np


def parse_hamiltonian_file(filename: str) -> dict:
    hamil_info = {}
    with open(filename, 'r') as rfid:
        lines = rfid.read().split('\n')
        hamil_list = []
        for k, line in enumerate(lines):
            if not line.strip():
                continue
            line = line.strip()
            hamil = {'wires': [], 'observables': []}
            if k == 0:
                name, method, n_wires = line.split(' ')
                hamil_info['name'] = name
                hamil_info['method'] = method
                hamil_info['n_wires'] = eval(n_wires)
            else:
                info = line.split(' ')
                hamil['coefficient'] = eval(info[0])
                for observable in info[1:]:
                    assert observable[0].lower() in ['x', 'y', 'z', 'i']
                    hamil['wires'].append(int(eval(observable[1:])))
                    hamil['observables'].append(observable[0].lower())
                hamil_list.append(hamil)

    hamil_info['hamil_list'] = hamil_list

    return hamil_info


def generate_n_hamiltonian(n_wires: int, n_hamil: int, n_lines: int) -> dict:
    assert 0 < n_hamil <= n_wires

    hamil_info = {'name': 'generated',
                  'method': f"{n_hamil}_hamil",
                  'n_wires': n_wires,
                  'hamil_list': []}

    combs = list(map(list, itertools.combinations(range(n_wires), n_hamil)))

    ctr_lines = 0
    while ctr_lines < n_lines:
        hamil = {}
        comb = combs[np.random.choice(len(combs))]
        comb.sort()
        hamil['wires'] = comb
        hamil['observables'] = list(np.random.choice(['x', 'y', 'z'], n_hamil))
        hamil['coefficient'] = 1.

        if hamil in hamil_info['hamil_list']:
            print(hamil)
            continue
        else:
            hamil_info['hamil_list'].append(hamil)
            ctr_lines += 1

    return hamil_info


def test_parse_hamiltonian_file():
    file = './examples/data/vqe/h2/h2.txt'
    print(parse_hamiltonian_file(file))


def test_generate_n_hamiltonian():
    print(generate_n_hamiltonian(n_wires=5, n_hamil=3, n_lines=100))


if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    test_parse_hamiltonian_file()
    test_generate_n_hamiltonian()
