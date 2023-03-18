# import torch
# import torchquantum as tq
# import numpy as np
#
# from typing import Any
#
#
# class QuantizeFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx: Any, x: torch.Tensor) -> Any:
#         # should be round so that the changes would be small, values close to
#         # 2pi should go to 2pi
#         return x.round()
#
#     @staticmethod
#     def backward(ctx: Any, grad_output: Any) -> Any:
#         grad_input = grad_output.clone()
#         mean, std = grad_input.mean(), grad_input.std()
#         return grad_input.clamp_(mean - 3 * std, mean + 3 * std)
#
#
# class CliffordQuantizer(object):
#     def __init__(self):
#         pass
#
#     # straight-through estimator
#     @staticmethod
#     def quantize_sse(params):
#         param = params[0][0]
#         param = param % (2 * np.pi)
#         param = np.pi / 2 * QuantizeFunction.apply(param /
#                                                    (np.pi / 2))
#         params = param.unsqueeze(0).unsqueeze(0)
#         return params

import qiskit
import os
import shutil

if __name__ == "__main__":
    path = os.path.abspath(qiskit.__file__)
    print(path)
    # path for aer provider
    path_provider = path.replace("__init__.py", "providers/aer/backends/aerbackend.py")
    print(path_provider)
    fixed_file = "aerbackend_fixed.py"

    with open(path_provider, "r") as fid:
        for line in fid.readlines():
            if "FIXED" in line:
                print("The qiskit parameterization bug is already fixed!")
                exit(0)
            else:
                print(
                    f"Fixing the qiskit parameterization bug by replacing "
                    f"the {path_provider} file with {fixed_file}!"
                )
                break

    shutil.copyfile(
        path_provider, path_provider.replace("aerbackend", "aerbackend.orig")
    )
    shutil.copyfile(fixed_file, path_provider)
    print("Finished!")
