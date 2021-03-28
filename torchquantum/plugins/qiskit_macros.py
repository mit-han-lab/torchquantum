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
              'ibmq_5_yorktown', 'ibmq_armonk']
