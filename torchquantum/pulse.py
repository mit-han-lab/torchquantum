import torch
import torch.nn as nn
import numpy as np

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
        return f"QuantumPulse Direct \n shape: {self.pulse_shape}"


class QuantumPulseGaussian(QuantumPulse):
    """Gaussian Quantum Pulse, will only count +- five sigmas
    """
    def __init__(self,
                 hamil,
                 n_steps: int = 100,
                 delta_t: float = 1.,
                 x_min: float = -10,
                 x_max: float = 10,
                 initial_params: List[float] = None,
                 ):
        super(QuantumPulseGaussian, self).__init__()
        self.hamil = torch.tensor(hamil, dtype=torch.complex64)
        self.delta_t = delta_t
        # mag, mu, sigma
        if initial_params is not None:
            assert len(initial_params) == 3
            initial_params = torch.Tensor(initial_params)
        else:
            initial_params = torch.ones(3)

        self.pulse_params = nn.Parameter(initial_params)
        self.n_steps = n_steps
        self.delta_x = (x_max - x_min) / n_steps
        self.x_list = torch.tensor(np.arange(x_min, x_max, self.delta_x))

    def get_unitary(self):
        self.mag = self.pulse_params[0]
        self.mu = self.pulse_params[1]
        self.sigma = self.pulse_params[2]

        # delta_x = (10 * self.sigma / self.n_steps).item()
        # self.x_list = torch.tensor(np.arange(
        # (self.mu - 5 * self.sigma).item(),
        # (self.mu + 5 * self.sigma).item(), delta_x))

        self.pulse_shape = self.mag * torch.exp(-(self.x_list - self.mu) ** 2 /
                                                (2 * self.sigma ** 2))

        unitary_per_step = []
        for k in range(self.n_steps):
            magnitude = self.pulse_shape[k]
            unitary = torch.matrix_exp(-1j * self.hamil * magnitude * self.delta_t * self.delta_x)
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
        return f"QuantumPulse Guassian \n shape: {self.pulse_shape}"


if __name__=='__main__':
    import pdb
    pdb.set_trace()
    pulse = QuantumPulseDirect(n_steps=10,
                               hamil=[[0, 1], [1, 0]])

    print(pulse.get_unitary())

    print("finish")
