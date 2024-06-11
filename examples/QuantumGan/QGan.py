import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchquantum as tq

class Generator(nn.Module):
    def __init__(self, n_qubits: int, latent_dim: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim

        # Quantum encoder
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [i], 'func': 'rx', 'wires': [i]}
            for i in range(self.n_qubits)
        ])

        # RX gates
        self.rxs = nn.ModuleList([
            tq.RX(has_params=True, trainable=True) for _ in range(self.n_qubits)
        ])

    def forward(self, x):
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)

        for i in range(self.n_qubits):
            self.rxs[i](qdev, wires=i)

        return tq.measure(qdev)

class Discriminator(nn.Module):
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits

        # Quantum encoder
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [i], 'func': 'rx', 'wires': [i]}
            for i in range(self.n_qubits)
        ])

        # RX gates
        self.rxs = nn.ModuleList([
            tq.RX(has_params=True, trainable=True) for _ in range(self.n_qubits)
        ])

        # Quantum measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)

        for i in range(self.n_qubits):
            self.rxs[i](qdev, wires=i)

        return self.measure(qdev)

class QGAN(nn.Module):
    def __init__(self, n_qubits: int, latent_dim: int):
        super().__init__()
        self.generator = Generator(n_qubits, latent_dim)
        self.discriminator = Discriminator(n_qubits)

    def forward(self, z):
        fake_data = self.generator(z)
        fake_output = self.discriminator(fake_data)
        return fake_output

def main(n_qubits, latent_dim):
    model = QGAN(n_qubits, latent_dim)
    print(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Generative Adversarial Network (QGAN) Example")
    parser.add_argument('n_qubits', type=int, help='Number of qubits')
    parser.add_argument('latent_dim', type=int, help='Dimension of the latent space')

    args = parser.parse_args()

    main(args.n_qubits, args.latent_dim)


{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Quantum Generative Adversarial Network (QGAN) Example\n",
    "\n",
    "In this notebook, we demonstrate the implementation of a Quantum Generative Adversarial Network (QGAN) using PyTorch and TorchQuantum.\n",
    "\n",
    "## Overview\n",
    "A QGAN consists of two main components:\n",
    "\n",
    "1. **Generator:** This network generates fake quantum data samples.\n",
    "2. **Discriminator:** This network tries to distinguish between real and fake quantum data samples.\n",
    "\n",
    "The goal is to train the generator to produce quantum data that is indistinguishable from real data, according to the discriminator. This is achieved through an adversarial training process, where the generator and discriminator are trained simultaneously in a competitive manner."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import torchquantum as tq\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Define the Generator class\n",
    "\n",
    "The Generator class defines a quantum circuit that takes a latent vector as input and generates a quantum state. This state is then measured to produce the generated data samples."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Generator(nn.Module):\n",
    "    def __init__(self, n_qubits: int, latent_dim: int):\n",
    "        super().__init__()\n",
    "        self.n_qubits = n_qubits\n",
    "        self.latent_dim = latent_dim\n",
    "\n",
    "        # Quantum encoder\n",
    "        self.encoder = tq.GeneralEncoder([\n",
    "            {'input_idx': [i], 'func': 'rx', 'wires': [i]}\n",
    "            for i in range(self.n_qubits)\n",
    "        ])\n",
    "\n",
    "        # RX gates\n",
    "        self.rxs = nn.ModuleList([\n",
    "            tq.RX(has_params=True, trainable=True) for _ in range(self.n_qubits)\n",
    "        ])\n",
    "\n",
    "    def forward(self, x):\n",
    "        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)\n",
    "        self.encoder(qdev, x)\n",
    "\n",
    "        for i in range(self.n_qubits):\n",
    "            self.rxs[i](qdev, wires=i)\n",
    "\n",
    "        return tq.measure(qdev)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Define the Discriminator class\n",
    "\n",
    "The Discriminator class defines a quantum circuit that takes data samples as input and tries to classify them as real or fake. The discriminator learns to distinguish between the real data and the data generated by the generator."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Discriminator(nn.Module):\n",
    "    def __init__(self, n_qubits: int):\n",
    "        super().__init__()\n",
    "        self.n_qubits = n_qubits\n",
    "\n",
    "        # Quantum encoder\n",
    "        self.encoder = tq.GeneralEncoder([\n",
    "            {'input_idx': [i], 'func': 'rx', 'wires': [i]}\n",
    "            for i in range(self.n_qubits)\n",
    "        ])\n",
    "\n",
    "        # RX gates\n",
    "        self.rxs = nn.ModuleList([\n",
    "            tq.RX(has_params=True, trainable=True) for _ in range(self.n_qubits)\n",
    "        ])\n",
    "\n",
    "        # Quantum measurement\n",
    "        self.measure = tq.MeasureAll(tq.PauliZ)\n",
    "\n",
    "    def forward(self, x):\n",
    "        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)\n",
    "        self.encoder(qdev, x)\n",
    "\n",
    "        for i in range(self.n_qubits):\n",
    "            self.rxs[i](qdev, wires=i)\n",
    "\n",
    "        return self.measure(qdev)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Define the QGAN class\n",
    "\n",
    "The QGAN class combines the generator and discriminator into a single model. The generator produces fake data, and the discriminator evaluates this data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class QGAN(nn.Module):\n",
    "    def __init__(self, n_qubits: int, latent_dim: int):\n",
    "        super().__init__()\n",
    "        self.generator = Generator(n_qubits, latent_dim)\n",
    "        self.discriminator = Discriminator(n_qubits)\n",
    "\n",
    "    def forward(self, z):\n",
    "        fake_data = self.generator(z)\n",
    "        fake_output = self.discriminator(fake_data)\n",
    "        return fake_output\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Define the main function\n",
    "\n",
    "The main function initializes the QGAN model and prints it. This function can be called with different numbers of qubits and latent dimensions to create and examine various configurations of the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def main(n_qubits, latent_dim):\n",
    "    model = QGAN(n_qubits, latent_dim)\n",
    "    print(model)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Use interact to dynamically create and display the model\n",
    "\n",
    "We use `interact` from `ipywidgets` to create an interactive interface for adjusting the number of qubits and the latent dimension. This allows you to see how the QGAN model changes with different parameters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from ipywidgets import interact, IntSlider\n",
    "\n",
    "@interact(n_qubits=IntSlider(min=1, max=10, step=1, value=4), latent_dim=IntSlider(min=1, max=10, step=1, value=2))\n",
    "def create_model(n_qubits, latent_dim):\n",
    "    model = QGAN(n_qubits, latent_dim)\n",
    "    print(model)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
