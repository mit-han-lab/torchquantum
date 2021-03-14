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

