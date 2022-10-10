__version__ = "0.1.4"
__author__ = 'Hanrui Wang'

from .macro import *
from .devices import *
from .module import *
from .operators import *
from .measurement import *
from .functional import *
from .graph import *
from .layers import *
from .encoding import *
from .super_layers import *
from .utils import *
from .noise_model import *
from .node import *
from .states import *
from .pulse import *
# from .pulse_utils import *

# here we check whether the Qiskit parameterization bug is fixed, if not, a
# warning message will be printed
import qiskit
import os
path = os.path.abspath(qiskit.__file__)
# print(path)
# path for aer provider
path_provider = path.replace('__init__.py',
                             'providers/aer/backends/aerbackend.py')
# print(path_provider)

# with open(path_provider, 'r') as fid:
#     for line in fid.readlines():
#         if 'FIXED' in line:
#             # print('The qiskit parameterization bug is already fixed!')
#             break
#         else:
#             print(f'\n\n WARNING: The qiskit parameterization bug is not '
#                   f'fixed!\n\n'
#                   f'run python fix_qiskit_parameterization.py to fix it!'
#                   )
#             break
