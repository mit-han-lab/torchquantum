import torchquantum as tq

QISKIT_INCOMPATIBLE_OPS = [
    tq.Rot,
    tq.MultiRZ,
    tq.CRot,
]

QISKIT_INCOMPATIBLE_FUNC_NAMES = [
    'rot',
    'multirz',
    'crot',
]

IBMQ_NAMES = ['ibmq_santiago', 'ibmq_athens', 'ibmq_belem',
              'ibmq_quito', 'ibmq_16_melbourne', 'ibmq_lima',
              'ibmq_5_yorktown', 'ibmq_armonk', 'ibmqx2', 'ibmq_montreal',
              'ibmq_kolkata', 'ibmq_mumbai', 'ibmq_dublin', 'ibmq_manhattan',
              'ibmq_paris', 'ibmq_toronto', 'ibmq_sydney', 'ibmq_guadalupe',
              'ibmq_casablanca', 'ibmq_bogota',
              'ibmq_rome', 'ibmq_manila',
              'ibmq_jakarta',
              'simulator_stabilizer',
              'simulator_mps',
              'simulator_extended_stabilizer',
              'simulator_statevector',
              'ibmq_qasm_simulator',
              ]

IBMQ_PNAMES = ['FakeQuito', 'FakeArmonk', 'FakeBogota']
