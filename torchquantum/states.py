import torch
import torch.nn as nn
import numpy as np
import torchquantum.functional as tqf


from torchquantum.macro import C_DTYPE
from typing import Union, List, Iterable


__all__ = ['QuantumState']


class QuantumState(nn.Module):
    """
    The Quantum Statevector.
    """

    def __init__(self, n_wires: int,
                 bsz: int = 1):
        """Init function for QuantumState class
        Args:
            n_wires (int): how many qubits for the state.
            bsz (int): batch size.
        """
        super().__init__()
        # number of qubits
        # the states are represented in a multi-dimension tensor
        # from left to right: qubit 0 to n
        self.n_wires = n_wires

        _state = torch.zeros(2 ** self.n_wires, dtype=C_DTYPE)
        _state[0] = 1 + 0j
        _state = torch.reshape(_state, [2] * self.n_wires)
        self.register_buffer('state', _state)

        repeat_times = [bsz] + [1] * len(self.state.shape)
        self._states = self.state.repeat(*repeat_times)
        self.register_buffer('states', self._states)

        self.op_list = []

    def clone_states(self, existing_states: torch.Tensor):
        self.states = existing_states.clone()

    def set_states(self, states: Union[torch.Tensor, List]):
        states = torch.tensor(states, dtype=C_DTYPE).to(self.state.device)
        bsz = states.shape[0]
        self.states = torch.reshape(states, [bsz] + [2] * self.n_wires)

    def reset_states(self, bsz: int):
        repeat_times = [bsz] + [1] * len(self.state.shape)
        self.states = self.state.repeat(*repeat_times).to(self.state.device)

    def reset_identity_states(self):
        """Make the states as the identity matrix, one dim is the batch
        dim. Useful for verification.
        """
        self.states = torch.eye(2 ** self.n_wires, device=self.state.device,
                                dtype=C_DTYPE).reshape([2 ** self.n_wires] +
                                                       [2] * self.n_wires)

    def reset_all_eq_states(self, bsz: int):
        energy = np.sqrt(1 / (2 ** self.n_wires) / 2)
        all_eq_state = torch.ones(2 ** self.n_wires, dtype=C_DTYPE) * \
            (energy + energy * 1j)
        all_eq_state = all_eq_state.reshape([2] * self.n_wires)
        repeat_times = [bsz] + [1] * len(self.state.shape)
        self.states = all_eq_state.repeat(*repeat_times).to(self.state.device)

    def get_states_1d(self):
        bsz = self.states.shape[0]
        return torch.reshape(self.states, [bsz, 2 ** self.n_wires])

    def get_state_1d(self):
        return torch.reshape(self.state, [2 ** self.n_wires])

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.name} {self.n_wires} wires \n state: {self.get_states_1d()}"

    def hadamard(self,
                 wires: Union[List[int], int],
                 inverse: bool = False,
                 comp_method: str = 'bmm'):
        tqf.hadamard(self,
                     wires=wires,
                     inverse=inverse,
                     comp_method=comp_method)

    def shadamard(self,
                  wires: Union[List[int], int],
                  inverse: bool = False,
                  comp_method: str = 'bmm'):
        tqf.shadamard(self,
                      wires=wires,
                      inverse=inverse,
                      comp_method=comp_method)

    def paulix(self,
               wires: Union[List[int], int],
               inverse: bool = False,
               comp_method: str = 'bmm'):
        tqf.paulix(self,
                   wires=wires,
                   inverse=inverse,
                   comp_method=comp_method)

    def pauliy(self,
               wires: Union[List[int], int],
               inverse: bool = False,
               comp_method: str = 'bmm'):
        tqf.pauliy(self,
                   wires=wires,
                   inverse=inverse,
                   comp_method=comp_method)

    def pauliz(self,
               wires: Union[List[int], int],
               inverse: bool = False,
               comp_method: str = 'bmm'):
        tqf.pauliz(self,
                   wires=wires,
                   inverse=inverse,
                   comp_method=comp_method)

    def i(self,
          wires: Union[List[int], int],
          inverse: bool = False,
          comp_method: str = 'bmm'):
        tqf.i(self,
              wires=wires,
              inverse=inverse,
              comp_method=comp_method)

    def s(self,
          wires: Union[List[int], int],
          inverse: bool = False,
          comp_method: str = 'bmm'):
        tqf.s(self,
              wires=wires,
              inverse=inverse,
              comp_method=comp_method)

    def t(self,
          wires: Union[List[int], int],
          inverse: bool = False,
          comp_method: str = 'bmm'):
        tqf.t(self,
              wires=wires,
              inverse=inverse,
              comp_method=comp_method)

    def sx(self,
           wires: Union[List[int], int],
           inverse: bool = False,
           comp_method: str = 'bmm'):
        tqf.sx(self,
               wires=wires,
               inverse=inverse,
               comp_method=comp_method)

    def cnot(self,
             wires: Union[List[int], int],
             inverse: bool = False,
             comp_method: str = 'bmm'):
        tqf.cnot(self,
                 wires=wires,
                 inverse=inverse,
                 comp_method=comp_method)

    def cz(self,
           wires: Union[List[int], int],
           inverse: bool = False,
           comp_method: str = 'bmm'):
        tqf.cz(self,
               wires=wires,
               inverse=inverse,
               comp_method=comp_method)

    def cy(self,
           wires: Union[List[int], int],
           inverse: bool = False,
           comp_method: str = 'bmm'):
        tqf.cy(self,
               wires=wires,
               inverse=inverse,
               comp_method=comp_method)

    def swap(self,
             wires: Union[List[int], int],
             inverse: bool = False,
             comp_method: str = 'bmm'):
        tqf.swap(self,
                 wires=wires,
                 inverse=inverse,
                 comp_method=comp_method)

    def sswap(self,
              wires: Union[List[int], int],
              inverse: bool = False,
              comp_method: str = 'bmm'):
        tqf.sswap(self,
                  wires=wires,
                  inverse=inverse,
                  comp_method=comp_method)

    def cswap(self,
              wires: Union[List[int], int],
              inverse: bool = False,
              comp_method: str = 'bmm'):
        tqf.cswap(self,
                  wires=wires,
                  inverse=inverse,
                  comp_method=comp_method)

    def toffoli(self,
                 wires: Union[List[int], int],
                 inverse: bool = False,
                 comp_method: str = 'bmm'):
        tqf.toffoli(self,
                     wires=wires,
                     inverse=inverse,
                     comp_method=comp_method)

    def multicnot(self,
                wires: Union[List[int], int],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        tqf.multicnot(self,
                    wires=wires,
                    inverse=inverse,
                    comp_method=comp_method)

    def multixcnot(self,
                wires: Union[List[int], int],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        tqf.multixcnot(self,
                    wires=wires,
                    inverse=inverse,
                    comp_method=comp_method)

    def rx(self,
           wires: Union[List[int], int],
           params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
           inverse: bool = False,
           comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.rx(self,
               wires=wires,
               params=params,
               inverse=inverse,
               comp_method=comp_method)

    def ry(self,
           wires: Union[List[int], int],
           params: torch.Tensor,
           inverse: bool = False,
           comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.ry(self,
               wires=wires,
               params=params,
               inverse=inverse,
               comp_method=comp_method)

    def rz(self,
           wires: Union[List[int], int],
           params: torch.Tensor,
           inverse: bool = False,
           comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.rz(self,
               wires=wires,
               params=params,
               inverse=inverse,
               comp_method=comp_method)

    def rxx(self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.rxx(self,
                wires=wires,
                params=params,
                inverse=inverse,
                comp_method=comp_method)

    def ryy(self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.ryy(self,
                wires=wires,
                params=params,
                inverse=inverse,
                comp_method=comp_method)

    def rzz(self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.rzz(self,
                wires=wires,
                params=params,
                inverse=inverse,
                comp_method=comp_method)

    def rzx(self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.rzx(self,
                wires=wires,
                params=params,
                inverse=inverse,
                comp_method=comp_method)

    def phaseshift(self,
                   wires: Union[List[int], int],
                   params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                   inverse: bool = False,
                   comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.phaseshift(self,
                       wires=wires,
                       params=params,
                       inverse=inverse,
                       comp_method=comp_method)

    def rot(self,
            wires: Union[List[int], int],
            params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
            inverse: bool = False,
            comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        tqf.rot(self,
                wires=wires,
                params=params,
                inverse=inverse,
                comp_method=comp_method)

    def multirz(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.multirz(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def crx(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.crx(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def cry(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.cry(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def crz(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.crz(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def crot(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        tqf.crot(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def u1(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.u1(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def u2(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        tqf.u2(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def u3(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        tqf.u3(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def cu1(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        tqf.cu1(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def cu2(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        tqf.cu2(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def cu3(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        if params.dim() == 1:
            params = params.unsqueeze(0)

        tqf.cu3(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def qubitunitary(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.tensor(params, dtype=C_DTYPE)
        else:
            params = torch.tensor([params], dtype=C_DTYPE)

        tqf.qubitunitary(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def qubitunitaryfast(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.tensor(params, dtype=C_DTYPE)
        else:
            params = torch.tensor([params], dtype=C_DTYPE)

        tqf.qubitunitaryfast(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def qubitunitarystrict(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.tensor(params, dtype=C_DTYPE)
        else:
            params = torch.tensor([params], dtype=C_DTYPE)

        tqf.qubitunitarystrict(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    def single_excitation(self,
                wires: Union[List[int], int],
                params: Union[torch.Tensor, np.ndarray, List[float], List[int], int, float],
                inverse: bool = False,
                comp_method: str = 'bmm'):

        if isinstance(params, Iterable):
            params = torch.Tensor(params)
        else:
            params = torch.Tensor([params])

        tqf.single_excitation(self,
                    wires=wires,
                    params=params,
                    inverse=inverse,
                    comp_method=comp_method)

    h = hadamard
    sh = shadamard
    x = paulix
    y = pauliy
    z = pauliz
    xx = rxx
    yy = ryy
    zz = rzz
    zx = rzx
    cx = cnot
    ccnot = toffoli
    ccx = toffoli
    u = u3
    cu = cu3
    p = phaseshift
    cp = cu1
    cr = cu1
    cphase = cu
