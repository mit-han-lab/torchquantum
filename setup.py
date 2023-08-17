from setuptools import setup, find_packages

VERSION = {}  # type: ignore

with open("torchquantum/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)

if __name__ == "__main__":
    requirements = open("requirements.txt").readlines()
    requirements = [r.strip() for r in requirements]

    setup(
        name="torchquantum",
        version=VERSION["version"],
        description="A PyTorch-based framework for differentiable classical simulation of quantum computing",
        url="https://github.com/mit-han-lab/torchquantum",
        author="Hanrui Wang, Jiannan Cao, Jessica Ding, Jiai Gu, Song Han, Zirui Li, Zhiding Liang, Pengyu Liu, Mohammadreza Tavasoli",
        author_email="hanruiwang.hw@gmail.com",
        license="MIT",
        install_requires=requirements,
        extras_require={"doc": ["nbsphinx", "recommonmark"]},
        python_requires=">=3.5",
        include_package_data=True,
        packages=find_packages(),
    )
