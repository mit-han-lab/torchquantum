import numpy as np
import torch
import cmath

def sigmai():
    i_value = np.array([[1,0],[0,1]])
    return torch.tensor(i_value,dtype=torch.complex64)

def sigmax():
    x_value =  np.array([[0,1],[1,0]])
    return torch.tensor(x_value,dtype=torch.complex64)

def sigmay():
    y_value = np.array([[0,-1.j],[1.j,0]])
    return torch.tensor(y_value,dtype=torch.complex64)

def sigmaz():
    z_value = np.array([[1,0],[0,-1]])
    return torch.tensor(z_value,dtype=torch.complex64)

def sigmaplus():
    p_value = np.array([[0,2],[0,0]])
    return torch.tensor(p_value,dtype=torch.complex64)

def sigmaminus():
    m_value = np.array([[0,0],[2,0]])
    return torch.tensor(m_value,dtype=torch.complex64)

def InitialState(n_qubit = 1, state = [0]):
    assert len(state) == n_qubit
    active_ind = 0
    for element in reversed(state):
        assert element==0 or element==1
        active_ind = (active_ind << 1) | element
    state_length = 2**n_qubit
    initial_value = np.zeros(state_length,dtype=int)
    initial_value[active_ind] = 1
    initial_state = torch.tensor(initial_value,dtype=torch.complex64)
    return initial_state

def InitialDensity(n_qubit = 1, state = [0]):
    initial_state = InitialState(n_qubit, state)
    initial_density = torch.ger(initial_state, torch.conj(initial_state))
    return initial_density

def H_2q_example(pulse, dt):
    def H(t):
        t_ind = (t/dt).long()
        interaction = -1.548*torch.kron(sigmai(),sigmax())-0.004*torch.kron(sigmai(),sigmay())-0.006*torch.kron(sigmai(),sigmaz()) \
        +5.316*torch.kron(sigmaz(),sigmax())-0.225*torch.kron(sigmaz(),sigmay())-0.340*torch.kron(sigmaz(),sigmaz())
        if t_ind >= len(pulse):
            return 0 * interaction
        return pulse[t_ind] * interaction + 1.225*torch.kron(sigmaz(),sigmai())
    return H

def H_qubit_example(n_qubit, pulse, dt):
    def H(t):
        t_ind = (t/dt).long()
        if t_ind >= len(pulse):
            return sigmax() * 0
        return sigmax()*pulse[t_ind].real + sigmay()*pulse[t_ind].imag
    return H

def H_larmor_example(n_qubit, pulse, dt):
    def H(t):
        t_ind = (t/dt).long()
        if t_ind>=len(pulse):
            return -sigmaz()*0.1 + sigmax() * 0
        return -sigmaz()*0.1 + sigmax()*pulse[t_ind]
    return H

def normalize_state_vector(a, b):
    magnitude = cmath.sqrt(a * a.conjugate() + b * b.conjugate())
    alpha = a / magnitude
    beta = b / magnitude
    return alpha, beta

def sv2bloch(sv):
    a = sv[0]
    b = sv[1]
    # alpha, beta = normalize_state_vector(a,b)
    # Compute x, y, and z coordinates
    x = 2 * np.real(b * np.conj(a))
    y = 2 * np.imag(b * np.conj(a))
    z = np.abs(a)**2 - np.abs(b)**2
    return [x, y, z]

def dens2bloch(rho):
    x = np.trace(np.dot(rho, sigmax()))
    y = np.trace(np.dot(rho, sigmay()))
    z = np.trace(np.dot(rho, sigmaz()))
    return [x.real, y.real, z.real]

def Schedule(pulse_list):
    pulse_value = torch.tensor(pulse_list,dtype=torch.complex64)
    pulse = torch.nn.parameter.Parameter(pulse_value)
    return pulse
