from setuptools import setup, find_packages

VERSION = {}  # type: ignore

with open("torchquantum/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)

if __name__ == "__main__":
    setup(
        name="torchquantum",
        version=VERSION["version"],
        description="A PyTorch-based framework for differentiable classical simulation of quantum computing",
        url="https://github.com/mit-han-lab/torchquantum",
        author="Hanrui Wang, Jiannan Cao, Jessica Ding, Jiai Gu, Song Han, Zirui Li, Zhiding Liang, Pengyu Liu, Mohammadreza Tavasoli",
        author_email="hanruiwang.hw@gmail.com",
        license="MIT",
        install_requires=[
            "numpy>=1.19.2",
            "torchvision>=0.9.0.dev20210130",
            "tqdm>=4.56.0",
            "setuptools>=52.0.0",
            "torch>=1.8.0",
            "torchpack>=0.3.0",
            "qiskit==0.38.0",
            "matplotlib>=3.3.2",
            "pathos>=0.2.7",
            "pylatexenc>=2.10",
            "dill==0.3.4",
        ],
        extras_require={"doc": ["nbsphinx", "recommonmark"]},
        python_requires=">=3.5",
        include_package_data=True,
        packages=find_packages(),
    )
