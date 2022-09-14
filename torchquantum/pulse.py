import torch
import torch.nn as nn
import numpy as np
import torchquantum.functional as tqf


from torchquantum.macro import C_DTYPE
from typing import Union, List, Iterable


__all__ = [
    'QuantumPulse',
    'QuantumPulseGaussian',
    'QuantumPulseDirect',
]


class QuantumPulse(nn.Module):
    """
    The Quantum Pulse simulator
    """
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


class QuantumPulseGaussian(QuantumPulse):
    def __init__(self):
        super().__init__()
        pass


class QuantumPulseDirect(QuantumPulse):
    def __init__(self,
                 n_steps: int,
                 hamil,
                 delta_t: float = 1.,
                 initial_shape: List[float] = None):
        super().__init__()
        self.hamil = torch.tensor(hamil, dtype=torch.complex64)
        if initial_shape is not None:
            assert len(initial_shape) == n_steps
            initial_shape = torch.Tensor(initial_shape)
        else:
            initial_shape = torch.ones(n_steps)
        self.pulse_shape = nn.Parameter(initial_shape)
        self.n_steps = n_steps
        self.delta_t = delta_t

    def get_unitary(self):
        unitary_per_step = []
        for k in range(self.n_steps):
            magnitude = self.pulse_shape[k]
            unitary = torch.matrix_exp(-1j * self.hamil * magnitude * self.delta_t)
            # print(unitary @ unitary.conj().T)
            # unitary_mag = (unitary[0]**2).sum().sqrt()
            # unitary = unitary_mag / unitary

            unitary_per_step.append(unitary)

        u_overall = None
        for k, u in enumerate(unitary_per_step):
            if not k:
                u_overall = u
            else:
                u_overall = u_overall @ u

        return u_overall

    def __repr__(self):
        return f"QuantumPulse {self.name} \n shape: {self.pulse_shape}"


if __name__=='__main__':
    import pdb
    pdb.set_trace()
    pulse = QuantumPulseDirect(n_steps=10,
                               hamil=[[0, 1], [1, 0]])

    print(pulse.get_unitary())

    print("finish")
