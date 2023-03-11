import math
import torch
import numpy as np

from string import ascii_lowercase

torch_numpy_dtype_dict = {
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
}

C_DTYPE = torch.complex64
F_DTYPE = torch.float32

ABC = ascii_lowercase
ABC_ARRAY = np.array(list(ABC))

INV_SQRT2 = 1 / math.sqrt(2)

C_DTYPE_NUMPY = torch_numpy_dtype_dict[C_DTYPE]
F_DTYPE_NUMPY = torch_numpy_dtype_dict[F_DTYPE]
