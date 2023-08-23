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

import torch
import torch.nn as nn
import numpy as np

from typing import Union, List, Iterable


__all__ = [
    "QuantumPulse",
    "QuantumPulseGaussian",
    "QuantumPulseDirect",
]


class QuantumPulse(nn.Module):
    """The Quantum Pulse simulator
    
    Methods:
        __init__(self):
            Initialize the QuantumPulse.
        forward(self)
    """

    def __init__(self):
        """Initialize the Quantum Pulse simulator.
        
        Returns:
            None.
        """
        
        super().__init__()
        pass

    def forward(self):
        pass


class QuantumPulseDirect(QuantumPulse):
    def __init__(
        self,
        n_steps: int,
        hamil,
        delta_t: float = 1.0,
        initial_shape: List[float] = None,
    ):
        """Initializes a QuantumPulseDirect object.

        Args:
            n_steps (int): The number of time steps.
            hamil: The Hamiltonian.
            delta_t (float, optional): The time step size.
                Defaults to 1.0.
            initial_shape (List[float], optional): The initial shape of the pulse. 
                Defaults to None.

        Raises:
            AssertionError: If the length of initial_shape is not equal to n_steps.

        Example:
            >>> hamiltonian = ...
            >>> pulse = QuantumPulseDirect(n_steps=10, hamil=hamiltonian, delta_t=0.1)
        """
        
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
        """Computes the unitary evolution operator for the pulse.

        Returns:
            torch.Tensor: The unitary evolution operator.

        Example:
            >>> pulse = QuantumPulseDirect(n_steps=10, hamil=hamiltonian, delta_t=0.1)
            >>> unitary = pulse.get_unitary()
        """
        
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
        """Returns the string representation of the QuantumPulseDirect object.

        Returns:
            str: The string representation.

        Example:
            >>> pulse = QuantumPulseDirect(n_steps=10, hamil=hamiltonian, delta_t=0.1)
            >>> print(pulse)
            QuantumPulse Direct
            shape: tensor([...])
        """
        
        return f"QuantumPulse Direct \n shape: {self.pulse_shape}"


class QuantumPulseGaussian(QuantumPulse):
    """Gaussian Quantum Pulse, will only count +- five sigmas"""

    def __init__(
        self,
        hamil,
        n_steps: int = 100,
        delta_t: float = 1.0,
        x_min: float = -10,
        x_max: float = 10,
        initial_params: List[float] = None,
    ):
        """Initializes a QuantumPulseGaussian object.

        Args:
            hamil: The Hamiltonian.
            n_steps (int, optional): The number of time steps.
                Defaults to 100.
            delta_t (float, optional): The time step size.
                Defaults to 1.0.
            x_min (float, optional): The minimum value of x.
                Defaults to -10.
            x_max (float, optional): The maximum value of x.
                Defaults to 10.
            initial_params (List[float], optional): The initial parameters of the pulse. 
                Defaults to None.

        Returns:
            None.

        Raises:
            AssertionError: If the length of initial_params is not equal to 3.

        Example:
            >>> hamiltonian = ...
            >>> pulse = QuantumPulseGaussian(hamil=hamiltonian, n_steps=100, delta_t=0.1)
        """
        
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
        """Computes the unitary evolution operator for the pulse.

        Returns:
            torch.Tensor: The unitary evolution operator.

        Example:
            >>> pulse = QuantumPulseGaussian(hamil=hamiltonian, n_steps=100, delta_t=0.1)
            >>> unitary = pulse.get_unitary()
        """
        
        self.mag = self.pulse_params[0]
        self.mu = self.pulse_params[1]
        self.sigma = self.pulse_params[2]

        # delta_x = (10 * self.sigma / self.n_steps).item()
        # self.x_list = torch.tensor(np.arange(
        # (self.mu - 5 * self.sigma).item(),
        # (self.mu + 5 * self.sigma).item(), delta_x))

        self.pulse_shape = self.mag * torch.exp(
            -((self.x_list - self.mu) ** 2) / (2 * self.sigma**2)
        )

        unitary_per_step = []
        for k in range(self.n_steps):
            magnitude = self.pulse_shape[k]
            unitary = torch.matrix_exp(
                -1j * self.hamil * magnitude * self.delta_t * self.delta_x
            )
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
        """Returns the string representation of the QuantumPulseGaussian object.

        Returns:
            str: The string representation.

        Example:
            >>> pulse = QuantumPulseGaussian(hamil=hamiltonian, n_steps=100, delta_t=0.1)
            >>> print(pulse)
            QuantumPulse Guassian
            shape: tensor([...])
        """
        return f"QuantumPulse Guassian \n shape: {self.pulse_shape}"


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    pulse = QuantumPulseDirect(n_steps=10, hamil=[[0, 1], [1, 0]])

    print(pulse.get_unitary())

    print("finish")
