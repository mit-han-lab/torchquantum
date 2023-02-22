import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torchquantum as tq
import torchquantum.functional as tqf

class QEffGradModel(tq.QuantumModule):
    """
    This class implements the efficient gradient method for quantum circuits.
    Args:
        Hamiltonians (list of n Hs) where defined as:
            
            H = [[Coeffcients], [Pauli Strings]]
            Here, H1 = $\sum_k beta_k Q_k^(1), ..., H_n = \sum_k beta_k Q_k^(3)$ where $Q_k$ is a Pauli

        parameters (list of floats): parameters for the unitaries defined by 
        U1 = $exp(-i theta1 H1/2 ), ..., Un = exp(-i theta2 H2/2 )$

        Observable (list of size two of coefficients and Pauli Strings): 
            E.g. Observable = [Coeffcients, [Pauli Strings]]
            Here, Observable(O) = \sum_l alpha_l P_l where P_l is a Pauli

    Returns:
        gradient (list of floats): gradient of the expectation value w.r.t. the parameters
    """
    
    def __init__(self, Hamiltonians, Observable, n_wires):
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.Hamiltonians = Hamiltonians
        self.parameters = torch.nn.ParameterList([torch.nn.Parameter(torch.rand(1)) for i in range(len(Hamiltonians))])
        self.Observable = Observable

    def circuit(self, j, sign):
        """implements the quantum circuit for efficient gradient 

        Args:
            j (int): index of the parameter w.r.t. which the gradient is computed
            sign (+1/-1): sign of the exponent in the gradient formula
        """
        #first apply the unitaries U1, ..., Un 
        #***********Need a function called tq.Exponential_U()***********
        for i in range(len(self.Hamiltonians)):
            tqf.Exponential_U(self.q_device, self.Hamiltonians[i], self.parameters[i], wires=range(self.q_device.n_wires))

        #apply exp(+ipi/4 O) where O is the observable
        tqf.Exponential_U(self.Observable, sign*np.pi/4, wires=range(self.q_device.n_wires))

        #apply the unitaries U_{n}^{dagger},...., U_{j+1}^{dagger}
        for i in range(len(self.Hamiltonians)-1, j, -1):
            tqf.Exponential_U(self.Hamiltonians[i], -self.parameters[i], wires=range(self.q_device.n_wires))

        #measure the expectation value of the observable
        return tqf.expval_joint_analytical(self.q_device, self.Hamiltonians[j])

    def forward(self):
        """
        Returns gradient of the expectation value w.r.t. all the parameters
        """
        gradient = []
        for i in range(len(self.Hamiltonians)):
            gradient.append(self.forward(i, 1) - self.forward(i, -1))
        return gradient
