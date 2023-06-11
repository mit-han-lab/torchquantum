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
