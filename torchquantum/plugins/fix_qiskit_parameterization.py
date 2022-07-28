import qiskit
import os
import shutil

if __name__ == '__main__':
    path = os.path.abspath(qiskit.__file__)
    print(path)
    # path for aer provider
    path_provider = path.replace('__init__.py',
                                 'providers/aer/backends/aerbackend.py')
    print(path_provider)
    fixed_file = 'aerbackend_fixed.py'

    with open(path_provider, 'r') as fid:
        for line in fid.readlines():
            if 'FIXED' in line:
                print('The qiskit parameterization bug is already fixed!')
                exit(0)
            else:
                print(f'Fixing the qiskit parameterization bug by replacing '
                      f'the {path_provider} file with {fixed_file}!')
                break

    shutil.copyfile(path_provider, path_provider.replace('aerbackend',
                                                         'aerbackend.orig'))
    shutil.copyfile(fixed_file,
                    path_provider)
    print('Finished!')
