from setuptools import setup, find_packages

setup(name='torchquantum',
      version='0.1.2',
      description='A PyTorch-centric hybrid classical-quantum dynamic '
                  'neural networks framework.',
      url='https://github.com/mit-han-lab/torchquantum',
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
            'qiskit>=0.32.0',
            'matplotlib>=3.3.2',
            'pathos>=0.2.7',
            'pylatexenc>=2.10'
      ],
      extras_require = {
            'doc': [
                  'nbsphinx',
                  'recommonmark'
            ]
      },
      python_requires='>=3.5',
      include_package_data=True,
      packages=find_packages()
)
