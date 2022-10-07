# test the controlled unitary function





import torchquantum as tq
import torchquantum.functional as tqf

import pdb
pdb.set_trace()

flag = 8

if flag == 1:
    state = tq.QuantumState(n_wires=3)
    print(state)

    state.qubitunitaryfast(wires=0, params=[[0, 1], [1, 0]])
    state.qubitunitaryfast(wires=2, params=[[0, 1], [1, 0]])

    print(state)
    state.qubitunitaryfast(wires=[0, 2, 1],
                           params=([[1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0, 1, 0]]))

    print(state)

    state = tq.QuantumState(n_wires=3)
    print(state)

    gate1 = tq.QubitUnitaryFast(init_params=[[0, 1], [1, 0]],
                               n_wires=1,
                               wires=0
                               )
    gate1(state)

    gate2 = tq.QubitUnitaryFast(init_params=[[0, 1], [1, 0]],
                               n_wires=1,
                               wires=2
                               )

    gate2(state)
    print(state)

    gate3 = tq.QubitUnitaryFast(init_params=([[1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0, 1, 0]]),
                               n_wires=3,
                               wires=[0, 2, 1]
                               )

    ## use controlled gate to implement a cnot

    gate3(state)
    print(state)

if flag == 2:
    state = tq.QuantumState(n_wires=2)
    print(state)
    gate0 = tq.PauliX(n_wires=1, wires=0)
    gate = tq.QubitUnitaryFast.from_controlled_operation(
        op=tq.PauliX(),
        c_wires=0,
        t_wires=1,
        trainable=False)

    gate0(state)
    print(state)

    gate(state)
    print(state)

if flag == 3:
    state = tq.QuantumState(n_wires=2)
    print(state)
    gate0 = tq.PauliX(n_wires=1, wires=0)
    gate = tq.QubitUnitaryFast.fromControlledOperation(
        op=tq.RX(has_params=True, init_params=0.25),
        c_wires=0,
        t_wires=1,
        trainable=False)

    gate0(state)
    print(state)

    gate(state)
    print(state)


if flag == 4:
    state = tq.QuantumState(n_wires=3)
    print(state)
    gate0 = tq.PauliX(n_wires=1, wires=0)
    gate1 = tq.PauliX(n_wires=1, wires=1)
    gate = tq.QubitUnitaryFast.from_controlled_operation(
        op=tq.CNOT(),
        c_wires=0,
        t_wires=[1, 2],
        trainable=False)

    gate0(state)
    print(state)

    gate1(state)
    print(state)

    gate(state)
    print(state)


if flag == 5:
    state = tq.QuantumState(n_wires=5)
    print(state)
    gate0 = tq.PauliX(n_wires=1, wires=1)
    gate1 = tq.PauliX(n_wires=1, wires=0)
    gate2 = tq.PauliX(n_wires=1, wires=4)
    gate = tq.QubitUnitaryFast.from_controlled_operation(
        op=tq.CNOT(),
        c_wires=1,
        t_wires=[[0, 2], [4, 3]],
        trainable=False)

    gate0(state)
    print(state)

    gate1(state)
    print(state)

    gate2(state)
    print(state)

    gate(state)
    print(state)


if flag == 6:
    state = tq.QuantumState(n_wires=5)
    print(state)
    gate0 = tq.PauliX(n_wires=1, wires=0)
    gate1 = tq.PauliX(n_wires=1, wires=1)
    gate2 = tq.PauliX(n_wires=1, wires=2)
    gate3 = tq.PauliX(n_wires=1, wires=3)

    # gate2 = tq.PauliX(n_wires=1, wires=4)
    gate = tq.QubitUnitaryFast.from_controlled_operation(
        op=tq.Toffoli(),
        c_wires=[0, 1],
        t_wires=[2, 3, 4],
        trainable=False)

    gate0(state)
    print(state)

    # gate1(state)
    # print(state)

    gate2(state)
    print(state)

    gate3(state)
    print(state)

    gate(state)
    print(state)

if flag == 7:
    state = tq.QuantumState(n_wires=9)
    print(state)
    gate0 = tq.PauliX(n_wires=1, wires=0)
    gate1 = tq.PauliX(n_wires=1, wires=1)
    gate2 = tq.PauliX(n_wires=1, wires=2)
    gate3 = tq.PauliX(n_wires=1, wires=3)
    gate4 = tq.PauliX(n_wires=1, wires=4)
    gate5 = tq.PauliX(n_wires=1, wires=6)
    gate6 = tq.PauliX(n_wires=1, wires=7)

    # gate2 = tq.PauliX(n_wires=1, wires=4)
    gate = tq.QubitUnitaryFast.from_controlled_operation(
        op=tq.Toffoli(),
        c_wires=[0, 1, 2],
        t_wires=[[3, 4, 5], [6, 7, 8]],
        trainable=False)

    gate0(state)
    print(state)

    gate1(state)
    print(state)

    gate2(state)
    print(state)

    gate3(state)
    print(state)

    gate4(state)
    print(state)

    gate5(state)
    print(state)

    gate6(state)
    print(state)

    gate(state)
    print(state)


if flag == 8:
    state = tq.QuantumState(n_wires=3)
    print(state)
    gate0 = tq.PauliX(n_wires=1, wires=0)
    gate1 = tq.PauliX(n_wires=1, wires=1)
    gate_cx = tq.QubitUnitaryFast.from_controlled_operation(
        op=tq.PauliX(),
        c_wires=0,
        t_wires=1,
        trainable=False)

    gate_ccx = tq.QubitUnitaryFast.from_controlled_operation(
        op=gate_cx,
        c_wires=0,
        t_wires=[1, 2],
        trainable=False
    )

    gate0(state)
    print(state)

    gate1(state)
    print(state)

    gate_ccx(state)
    print(state)