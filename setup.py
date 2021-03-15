from setuptools import setup, find_packages

setup(name='torchquantum',
      version='0.1.0',
      description='A PyTorch-centric hybrid classical-quantum dynamic '
                  'neural networks framework.',
      url='https://github.com/Hanrui-Wang/pytorch-quantum',
      author='Hanrui Wang',
      author_email='hanruiwang.hw@gmail.com',
      license='MIT',
      install_requires=[
            'numpy>=1.19.2',
            'torchvision>=0.9.0.dev20210130',
            'tqdm>=4.56.0',
            'setuptools>=52.0.0',
            'torch>=1.8.0',
            'torchquantum>=0.1',
            'torchpack>=0.3.0',
            'qiskit>=0.24.0',
            'matplotlib>=3.3.2'
      ],
      python_requires='>=3',
      include_package_data=True,
      packages=find_packages())
