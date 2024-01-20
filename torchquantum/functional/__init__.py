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

from .gate_wrapper import gate_wrapper, apply_unitary_einsum, apply_unitary_bmm
from .hadamard import hadamard, shadamard, _hadamard_mat_dict, h, ch, sh, chadamard
from .rx import rx, rxx, crx, xx, _rx_mat_dict, rx_matrix, rxx_matrix, crx_matrix
from .ry import ry, ryy, cry, yy, _ry_mat_dict, ry_matrix, ryy_matrix, cry_matrix
from .rz import (
    rz,
    rzz,
    crz,
    zz,
    zx,
    multirz,
    rzx,
    _rz_mat_dict,
    rz_matrix,
    rzz_matrix,
    crz_matrix,
    multirz_matrix,
    rzx_matrix,
)
from .phase_shift import phaseshift_matrix, phaseshift, p, _phaseshift_mat_dict
from .rot import rot, crot, rot_matrix, crot_matrix, _rot_mat_dict
from .reset import reset
from .xx_min_yy import xxminyy, xxminyy_matrix, _xxminyy_mat_dict
from .xx_plus_yy import xxplusyy, xxplusyy_matrix, _xxplusyy_mat_dict
from .u1 import u1, cu1, u1_matrix, cu1_matrix, _u1_mat_dict, cp, cr, cphase
from .u2 import u2, cu2, u2_matrix, cu2_matrix, _u2_mat_dict
from .u3 import u, u3, cu3, cu, cu_matrix, u3_matrix, cu3_matrix, _u3_mat_dict
from .qubit_unitary import (
    qubitunitary,
    qubitunitaryfast,
    qubitunitarystrict,
    qubitunitary_matrix,
    qubitunitaryfast_matrix,
    qubitunitarystrict_matrix,
    _qubitunitary_mat_dict,
)
from .single_excitation import (
    singleexcitation,
    singleexcitation_matrix,
    _singleexcitation_mat_dict,
)
from .paulix import (
    _x_mat_dict,
    multicnot_matrix,
    multixcnot_matrix,
    paulix,
    cnot,
    multicnot,
    multixcnot,
    x,
    c3x,
    c4x,
    dcx,
    toffoli,
    ccnot,
    ccx,
    cx,
    rccx,
    rc3x,
)
from .pauliy import _y_mat_dict, pauliy, cy, y
from .pauliz import _z_mat_dict, pauliz, cz, ccz, z
from .qft import _qft_mat_dict, qft, qft_matrix
from .r import _r_mat_dict, r, r_matrix
from .global_phase import _globalphase_mat_dict, globalphase, globalphase_matrix
from .sx import _sx_mat_dict, sx, c3sx, sxdg, csx
from .i import _i_mat_dict, i
from .s import _s_mat_dict, s, sdg, cs, csdg
from .t import _t_mat_dict, t, tdg
from .swap import _swap_mat_dict, swap, sswap, iswap, cswap
from .ecr import _ecr_mat_dict, ecr, echoedcrossresonance

mat_dict = {
    **_hadamard_mat_dict,
    **_rx_mat_dict,
    **_ry_mat_dict,
    **_rz_mat_dict,
    **_phaseshift_mat_dict,
    **_rot_mat_dict,
    **_xxminyy_mat_dict,
    **_xxplusyy_mat_dict,
    **_u1_mat_dict,
    **_u2_mat_dict,
    **_u3_mat_dict,
    **_qubitunitary_mat_dict,
    **_x_mat_dict,
    **_y_mat_dict,
    **_z_mat_dict,
    **_singleexcitation_mat_dict,
    **_qft_mat_dict,
    **_r_mat_dict,
    **_globalphase_mat_dict,
    **_sx_mat_dict,
    **_i_mat_dict,
    **_s_mat_dict,
    **_t_mat_dict,
    **_swap_mat_dict,
    **_ecr_mat_dict,
}

func_name_dict = {
    "hadamard": hadamard,
    "h": h,
    "sh": shadamard,
    "paulix": paulix,
    "pauliy": pauliy,
    "pauliz": pauliz,
    "i": i,
    "s": s,
    "t": t,
    "sx": sx,
    "cnot": cnot,
    "cz": cz,
    "cy": cy,
    "rx": rx,
    "ry": ry,
    "rz": rz,
    "rxx": rxx,
    "xx": xx,
    "ryy": ryy,
    "yy": yy,
    "rzz": rzz,
    "zz": zz,
    "rzx": rzx,
    "zx": zx,
    "swap": swap,
    "sswap": sswap,
    "cswap": cswap,
    "toffoli": toffoli,
    "phaseshift": phaseshift,
    "p": p,
    "cp": cp,
    "rot": rot,
    "multirz": multirz,
    "crx": crx,
    "cry": cry,
    "crz": crz,
    "crot": crot,
    "u1": u1,
    "u2": u2,
    "u3": u3,
    "u": u,
    "cu1": cu1,
    "cphase": cphase,
    "cr": cr,
    "cu2": cu2,
    "cu3": cu3,
    "cu": cu,
    "qubitunitary": qubitunitary,
    "qubitunitaryfast": qubitunitaryfast,
    "qubitunitarystrict": qubitunitarystrict,
    "multicnot": multicnot,
    "multixcnot": multixcnot,
    "x": x,
    "y": y,
    "z": z,
    "cx": cx,
    "ccnot": ccnot,
    "ccx": ccx,
    "reset": reset,
    "singleexcitation": singleexcitation,
    "ecr": ecr,
    "echoedcrossresonance": echoedcrossresonance,
    "qft": qft,
    "sdg": sdg,
    "tdg": tdg,
    "sxdg": sxdg,
    "ch": ch,
    "ccz": ccz,
    "iswap": iswap,
    "cs": cs,
    "csdg": csdg,
    "csx": csx,
    "chadamard": chadamard,
    "ccz": ccz,
    "dcx": dcx,
    "xxminyy": xxminyy,
    "xxplusyy": xxplusyy,
    "c3x": c3x,
    "r": r,
    "globalphase": globalphase,
    "c3sx": c3sx,
    "rccx": rccx,
    "rc3x": rc3x,
    "c4x": c4x,
}

from .func_mat_exp import matrix_exp
from .func_controlled_unitary import controlled_unitary

func_name_dict_collect = {
    "matrix_exp": matrix_exp,
    "controlled_unitary": controlled_unitary,
}
