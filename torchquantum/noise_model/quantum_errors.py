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

import numpy as np
import torch
import torchquantum as tq
from __future__ import annotations


class QuantumError(object):
    """
        A class for describing quantum error

    """

    def __init__(self):
        pass

    def compose(self, other: QuantumError) -> QuantumError:
        pass

    def tensor(self, other: QuantumError) -> QuantumError:
        pass

    def expand(self, other: QuantumError) -> QuantumError:
        pass


class krausError(QuantumError):
    """
        A general CPTP quantum error given a list of kraus matrices

        Params:

    """

    def __init__(self,
                 wires):
        pass


class mixed_unitary_error(QuantumError):
    """
        An n-qubit mixed unitary error

        Params:

    """

    def __init__(self,
                 wires):
        pass


class coherent_unitary_error(QuantumError):
    """
        An n qubit coheren error given a a single unitary

        Params:
    """

    def __init__(self,
                 wires):
        pass


class pauli_error(QuantumError):
    """
        An n qubit pauli error channel
        Initialized from a list of paulis and probability.

        Params:

        [(P0,p0),(P1,p1),...]

    """

    def __init__(self,
                 oplist):
        pass


class depolarizing_error(QuantumError):
    """
        An n qubit depolarizing error channel

        Params:

        Depolarization probability p

    """

    def __init__(self,
                 p):
        pass


class reset_error(QuantumError):
    """
        A single qubit reset error

        Params:
                p0:  Resetting of |0> state
                p1:  Resetting of |1> state

    """

    def __init__(self,
                 p0,
                 p1):
        pass


class thermal_relaxation_error(QuantumError):
    """
        A single qubit thermal relaxation channel.
        The relaxation process is determined by T1 and T2, gate time t,
        and excited state thermal population p1
        Params:
              T1:  Depolarization time
              T2:  Dephasing time
              t:   gate time
              p1:  excited state thermal population p1
    """

    def __init__(self,
                 T1,
                 T2,
                 t,
                 p1):
        pass


class phase_amplitude_damping_error(QuantumError):
    """
        A single qubit generalized combined phase and
        amplitude damping.

        Params:

        lambda: Amplitude damping parameter
        gamma: Phase damping parameter
        p1:  excited state thermal population p1

    """

    def __init__(self,
                 wires):
        pass


class amplitude_damping_error(QuantumError):
    """
        A class for describing amplitude damping

        Params:

        lambda: Amplitude damping parameter
        p1:  excited state thermal population p1

    """

    def __init__(self,
                 wires):
        pass


class phase_damping_error(QuantumError):
    """
        A class for describing quantum error

        Params:

        gamma: Phase damping parameter
        p1:  excited state thermal population p1

    """

    def __init__(self,
                 wires):
        pass
