import numpy as np
import os
# lib from Qiskit Aqua
# from qiskit.aqua import Operator, QuantumInstance
# from qiskit.aqua.algorithms import VQE, ExactEigensolver
# from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.operators import Z2Symmetries
from qiskit.circuit.instruction import Instruction
# lib from Qiskit Aqua Chemistry
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock

from torchquantum.plugins.qiskit_processor import QiskitProcessor
from torchquantum.plugins import qiskit2tq

from torchpack.utils.config import configs

processor = QiskitProcessor(
    use_real_qc=False,
    backend_name=None,
    noise_model_name=None,
    coupling_map_name=None,
    basis_gates_name=None,
    n_shots=8192,
    initial_layout=None,
    seed_transpiler=42,
    seed_simulator=42,
    optimization_level=None,
    max_jobs=5,
    remove_ops=False,
    remove_ops_thres=1e-4,
)

# import pdb
# pdb.set_trace()

def load_qubitop_for_molecule(molecule_data):
    atom_list = [a[0] + ' ' + " ".join([str(elem) for elem in a[1]]) for a in molecule_data['geometry']]
    atom = "; ".join(atom_list) 
    #atom = 'Li .0 .0 .0; H .0 .0 3.9'
    basis = molecule_data['basis']
    transform = molecule_data['transform']
    electrons = molecule_data['electrons']
    active = molecule_data['active_orbitals']
    driver = PySCFDriver(atom=atom, unit=UnitsType.ANGSTROM, basis=basis, charge=0, spin=0)
    molecule = driver.run()
    num_particles = molecule.num_alpha + molecule.num_beta
    num_spin_orbitals = molecule.num_orbitals * 2
    #print("# of electrons: {}".format(num_particles))
    #print("# of spin orbitals: {}".format(num_spin_orbitals))
    freeze_list = [x for x in range(int(active/2), int(num_particles/2))]
    remove_list = [-x for x in range(active,molecule.num_orbitals-int(num_particles/2)+int(active/2))]
    #print(freeze_list)
    #print(remove_list)

    if transform == 'BK':
        map_type = 'bravyi_kitaev'
    elif transform == 'JW':
        map_type = 'jordan_wigner'
    else:
        map_type = 'parity'
    remove_list = [x % molecule.num_orbitals for x in remove_list]
    freeze_list = [x % molecule.num_orbitals for x in freeze_list]
    remove_list = [x - len(freeze_list) for x in remove_list]
    remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
    freeze_list += [x + molecule.num_orbitals for x in freeze_list]
    fermiOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    energy_shift = 0
    if len(freeze_list) > 0:
        fermiOp, energy_shift = fermiOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
    if len(remove_list) > 0:
        fermiOp = fermiOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)
    qubitOp = fermiOp.mapping(map_type=map_type, threshold=0.00000001)
    if len(freeze_list) > 0 or len(remove_list) >0:
        qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)

    #print(qubitOp.print_operators())
    num_spin_orbitals= qubitOp.num_qubits
    return molecule, qubitOp, map_type, num_particles, num_spin_orbitals

def generate_uccsd(molecule_data):
    molecule, qubitOp, map_type, num_particles, num_spin_orbitals = load_qubitop_for_molecule(molecule_data)
    nuclear_repulsion_energy = molecule.nuclear_repulsion_energy

    print("# of electrons: {}".format(num_particles))
    print("# of spin orbitals: {}".format(num_spin_orbitals))
    qubit_reduction = False
    HF_state = HartreeFock(num_spin_orbitals, num_particles, map_type, qubit_reduction)
    uccsd_ansatz = UCCSD(reps=1,
                   num_orbitals=num_spin_orbitals, num_particles=num_particles,
                   initial_state=HF_state, qubit_mapping=map_type, 
                   two_qubit_reduction=qubit_reduction)
    circ = uccsd_ansatz.construct_circuit([0.4242] *
                                          uccsd_ansatz.num_parameters)
    circ.measure_all()
    circ_transpiled = processor.transpile(circ)

    q_layer = qiskit2tq(circ_transpiled)
    for name, param in q_layer.named_parameters():
        if not (param % (np.pi / 2)).detach().cpu().numpy().any():
            param.requires_grad = False

    #randlist = np.random.rand(uccsd_ansatz.num_parameters) # ansatz parameters
    #uccsd_ansatz_circuit = uccsd_ansatz.construct_circuit(randlist)
    if getattr(configs.model.arch, 'n_truncate_ops', None) is not None:
        # only take the front n_truncate_ops, otherwise cannot run on real QC
        q_layer.ops = q_layer.ops[:configs.model.arch.n_truncate_ops]

    return q_layer

def molecule_data2str(md):
    return md['name'] + ' ' + md['basis'] + ' ' + md['transform']+ ' ' + str(md['active_orbitals'])

def write_ansatz(molecule_data):
    #filename = ...
    ansatz = generate_uccsd(molecule_data)
    randlist = np.random.rand(uccsd_ansatz.num_parameters) # ansatz parameters
    uccsd_ansatz_circuit = uccsd_ansatz.construct_circuit(randlist)
    print(uccsd_ansatz_circuit)

def write_observable(molecule_data, root):
    #filename = ...
    _, qubitOp, _, _ , _ = load_qubitop_for_molecule(molecule_data)
    molecule_str = molecule_data2str(molecule_data)
    numq = qubitOp.num_qubits
    molecule_str += ' q' + str(numq) + '\n'
    op_str = qubitOp.print_details()
    filename = f"{molecule_data['name'].lower()}_" \
               f"{molecule_data['transform'].lower()}"
    with open(os.path.join(root, filename, f"{filename}.txt"), 'w') as wfid:
        wfid.write(f"{molecule_data['name'].lower()} "
                   f"{molecule_data['transform'].lower()} {numq}\n")
        for line in op_str.splitlines():
            molecule_str = ''
            #print(ord(line[6])) #ZXXIII (6.505213034913027e-19+0j)
            linedata = line.split(chr(9))
            if not complex(linedata[1]).imag == 0:
                print(f"WARNING: imaginary is not zero!!")

            molecule_str += str(complex(linedata[1]).real) + ' '
            for (i, c) in enumerate(linedata[0]):
                molecule_str += c+str(i)+' '

            wfid.write(f"{molecule_str}\n")

        # molecule_str
    # print(molecule_str)

        

    

# Molecule parameters for H2
h2_molecule = {
'name' : 'H2',
'basis' : 'sto-3g',
'transform' : 'BK',
'electrons' : 2,
'geometry' : [('H', (0., 0., 0.)), ('H', (0., 0., 0.72))],
'active_orbitals' : 2
}

# Molecule parameters for H2O
h2o_molecule = {
'name' : 'H2O',
'basis' : 'sto-3g',
'transform' : 'BK',
'electrons' : 8,
'geometry' : [('O', (0.,0.,0.)), ('H', (0.757,0.586,0.)), ('H', (-0.757,0.586,0.))],
'active_orbitals' : 4
}

# Molecule parameters for LiH
lih_molecule = {
'name' : 'LiH',
'basis' : 'sto-3g',
'transform' : 'BK',
'electrons' : 4,
'geometry' : [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))],
'active_orbitals' : 4
}

# Molecule parameters for CH4
ch4_molecule = {
'name' : 'CH4',
'basis' : 'sto-3g',
'transform' : 'BK',
'electrons' : 10,
'geometry' : [('C', (0, 0, 0)), ('H', (0.5541, 0.7996, 0.4965)), 
            ('H', (0.6833, -0.8134, -0.2536)), ('H', (-0.7782, -0.3735, 0.6692)), 
            ('H', (-0.4593, 0.3874, -0.9121))],
'active_orbitals' : 4
}

beh2_molecule = {
    'name' : 'BeH2',
    'basis' : 'sto-3g',
    'transform' : 'BK',
    'electrons' : 8,
    'geometry' : [('Be', (0., 0., 0.)), ('H', (0., 0., -1.7)), ('H', (0., 0., 1.7))],
    'active_orbitals' : 7
}

ch4new_molecule = {
    'name' : 'CH4NEW',
    'basis' : 'sto-3g',
    'transform' : 'BK',
    'electrons' : 10,
    'geometry' : [('C', (0, 0, 0)), ('H', (0.5541, 0.7996, 0.4965)),
                  ('H', (0.6833, -0.8134, -0.2536)), ('H', (-0.7782, -0.3735, 0.6692)),
                  ('H', (-0.4593, 0.3874, -0.9121))],
    'active_orbitals' : 6
}

#generate_uccsd(h2_molecule)
#generate_uccsd(h2o_molecule)
#generate_uccsd(lih_molecule)
#generate_uccsd(ch4_molecule)

molecule_name_dict = {
    'h2': h2_molecule,
    'h2o': h2o_molecule,
    'lih': lih_molecule,
    'ch4': ch4_molecule,
    'beh2': beh2_molecule,
    'ch4new': ch4new_molecule,
}

if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    # generate_uccsd(molecule_name_dict['ch4'])

# for transform in ['BK', 'JW']:
#     for name, info in molecule_name_dict.items():
#         root = './examples/data/vqe/'
#         info['transform'] = transform
#         os.makedirs(os.path.join(root, f"{name}_{transform.lower()}"),
#                     exist_ok=True)
#
#         write_observable(info, root)
