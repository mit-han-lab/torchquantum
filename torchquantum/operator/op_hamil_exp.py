import torch
from torchquantum.module import QuantumModule
from torchquantum.algorithm import Hamiltonian
import torchquantum.functional as tqf


__all__ = ['OpHamilExp']


class OpHamilExp(QuantumModule):
    """Matrix exponential operation.
    exp(-i * theta * H / 2)
    the default theta is 0.0
    """
    def __init__(self,
                 hamil: Hamiltonian,
                 trainable: bool = True,
                 theta: float = 0.0):
        """Initialize the OpHamilExp module.
        
        Args:
            hamil: The Hamiltonian.
            has_params: Whether the module has parameters.
            trainable: Whether the parameters are trainable.
            theta: The initial value of theta.
        
        """
        super().__init__()
        if trainable:
            self.theta = torch.nn.parameter.Parameter(torch.tensor(theta))
        else:
            self.theta = torch.tensor(theta)
        self.hamil = hamil
    
    def get_exponent_matrix(self):
        """Get the matrix on exponent."""
        return self.hamil.matrix * -1j * self.theta / 2
    
    @property
    def exponent_matrix(self):
        """Get the matrix on exponent."""
        return self.get_exponent_matrix()

    def get_matrix(self):
        """Get the overall matrix."""
        return torch.matrix_exp(self.exponent_matrix)
    
    @property
    def matrix(self):
        """Get the overall matrix."""
        return self.get_matrix()
    
    def forward(self, qdev, wires):
        """Forward the OpHamilExp module.
        Args:
            qdev: The QuantumDevice.
            wires: The wires.

        """
        matrix = self.matrix.to(qdev.device)
        tqf.qubitunitaryfast(
            q_device=qdev,
            wires=wires,
            params=matrix,
        )
