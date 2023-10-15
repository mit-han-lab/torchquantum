.. toctree::
   :maxdepth: 1
   :caption: API

   api_torchquantum
   api_functional
   api_operators
   api_layers

.. toctree::
   :maxdepth: 1
   :caption: Usage

   usage_installation
   examples/index.rst

.. raw:: html

   <embed>
      <h2> <p align="center"> A PyTorch Library for Quantum Simulation and Quantum Machine Learning</p> </h2>

      <h3> <p align="center"> Faster, Scalable, Easy Debugging, Easy Deployment on Real Machine</p> </h3>
    <p align="center">
    <a href="https://github.com/mit-han-lab/torchquantum/blob/master/LICENSE">
        <img alt="MIT License" src="https://img.shields.io/github/license/mit-han-lab/torchquantum">
    </a>
    <a href="https://torchquantum.readthedocs.io/">
        <img alt="Documentation" src="https://img.shields.io/readthedocs/torchquantum/main">
    </a>
    <a href="https://join.slack.com/t/torchquantum/shared_invite/zt-1ghuf283a-OtP4mCPJREd~367VX~TaQQ">
        <img alt="Chat @ Slack" src="https://img.shields.io/badge/slack-chat-2eb67d.svg?logo=slack">
    </a>
    <a href="https://qmlsys.hanruiwang.me">
        <img alt="Forum" src="https://img.shields.io/discourse/status?server=https%3A%2F%2Fqmlsys.hanruiwang.me%2F">
    </a>
    <a href="https://qmlsys.mit.edu">
        <img alt="Website" src="https://img.shields.io/website?up_message=qmlsys&url=https%3A%2F%2Fqmlsys.mit.edu">
    </a>

   <a href="https://pypi.org/project/torchquantum/">
        <img alt="Pypi" src="https://img.shields.io/pypi/v/torchquantum">
    </a>
      </p>
      <br />
    </embed>


👋 Welcome
==========

What it does
^^^^^^^^^^^^

Quantum simulation framework based on PyTorch. It supports statevector
simulation and pulse simulation (coming soon) on GPUs. It can scale up
to the simulation of 30+ qubits with multiple GPUs. 

Who will benefit
^^^^^^^^^^^^^^^^

Researchers on quantum algorithm design, parameterized quantum circuit
training, quantum optimal control, quantum machine learning, quantum
neural networks. 

Differences from Qiskit/Pennylane
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dynamic computation graph, automatic gradient computation, fast GPU
support, batch model tersorized processing.

News
----

-  v0.1.7 Available!
-  Join our
   `Slack <https://join.slack.com/t/torchquantum/shared_invite/zt-1ghuf283a-OtP4mCPJREd~367VX~TaQQ>`__
   for real time support!
-  Welcome to contribute! Please contact us or post in the
   `forum <https://qmlsys.hanruiwang.me>`__ if you want to have new
   examples implemented by TorchQuantum or any other questions.
-  Qmlsys website goes online:
   `qmlsys.mit.edu <https://qmlsys.mit.edu>`__ and
   `torchquantum.org <https://torchquantum.org>`__

Features
--------

-  Easy construction and simulation of quantum circuits in **PyTorch**
-  **Dynamic computation graph** for easy debugging
-  **Gradient support** via autograd
-  **Batch mode** inference and training on **CPU/GPU**
-  Easy **deployment on real quantum devices** such as IBMQ
-  **Easy hybrid classical-quantum** model construction
-  (coming soon) **pulse-level simulation**

Installation
------------

.. code:: bash

   git clone https://github.com/mit-han-lab/torchquantum.git
   cd torchquantum
   pip install --editable .

Basic Usage
-----------

.. code:: python

   import torchquantum as tq
   import torchquantum.functional as tqf

   qdev = tq.QuantumDevice(n_wires=2, bsz=5, device="cpu", record_op=True) # use device='cuda' for GPU

   # use qdev.op
   qdev.h(wires=0)
   qdev.cnot(wires=[0, 1])

   # use tqf
   tqf.h(qdev, wires=1)
   tqf.x(qdev, wires=1)

   # use tq.Operator
   op = tq.RX(has_params=True, trainable=True, init_params=0.5)
   op(qdev, wires=0)

   # print the current state (dynamic computation graph supported)
   print(qdev)

   # obtain the qasm string
   from torchquantum.plugin import op_history2qasm
   print(op_history2qasm(qdev.n_wires, qdev.op_history))

   # measure the state on z basis
   print(tq.measure(qdev, n_shots=1024))

   # obtain the expval on a observable by stochastic sampling (doable on simulator and real quantum hardware)
   from torchquantum.measurement import expval_joint_sampling
   expval_sampling = expval_joint_sampling(qdev, 'ZX', n_shots=1024)
   print(expval_sampling)

   # obtain the expval on a observable by analytical computation (only doable on classical simulator)
   from torchquantum.measurement import expval_joint_analytical
   expval = expval_joint_analytical(qdev, 'ZX')
   print(expval)

   # obtain gradients of expval w.r.t. trainable parameters
   expval[0].backward()
   print(op.params.grad)

.. raw:: html

   <!--
   ## Basic Usage 2

   ```python
   import torchquantum as tq
   import torchquantum.functional as tqf

   x = tq.QuantumDevice(n_wires=2)

   tqf.hadamard(x, wires=0)
   tqf.x(x, wires=1)
   tqf.cnot(x, wires=[0, 1])

   # print the current state (dynamic computation graph supported)
   print(x.states)

   # obtain the classical bitstring distribution
   print(tq.measure(x, n_shots=2048))
   ```
    -->

Guide to the examples
---------------------

We also prepare many example and tutorials using TorchQuantum.

For **beginning level**, you may check `QNN for
MNIST <examples/simple_mnist>`__, `Quantum Convolution
(Quanvolution) <examples/quanvolution>`__ and `Quantum Kernel
Method <examples/quantum_kernel_method>`__, and `Quantum
Regression <examples/regression>`__.

For **intermediate level**, you may check `Amplitude Encoding for
MNIST <examples/amplitude_encoding_mnist>`__, `Clifford gate
QNN <examples/clifford_qnn>`__, `Save and Load QNN
models <examples/save_load_example>`__, `PauliSum
Operation <examples/PauliSumOp>`__, `How to convert tq to
Qiskit <examples/converter_tq_qiskit>`__.

For **expert**, you may check `Parameter Shift on-chip
Training <examples/param_shift_onchip_training>`__, `VQA Gradient
Pruning <examples/gradient_pruning>`__, `VQE <examples/simple_vqe>`__,
`VQA for State Prepration <examples/train_state_prep>`__, `QAOA (Quantum
Approximate Optimization Algorithm) <examples/qaoa>`__.

Usage
-----

Construct parameterized quantum circuit models as simple as constructing
a normal pytorch model.

.. code:: python

   import torch.nn as nn
   import torch.nn.functional as F
   import torchquantum as tq
   import torchquantum.functional as tqf

   class QFCModel(nn.Module):
     def __init__(self):
       super().__init__()
       self.n_wires = 4
       self.measure = tq.MeasureAll(tq.PauliZ)

       self.encoder_gates = [tqf.rx] * 4 + [tqf.ry] * 4 + \
                            [tqf.rz] * 4 + [tqf.rx] * 4
       self.rx0 = tq.RX(has_params=True, trainable=True)
       self.ry0 = tq.RY(has_params=True, trainable=True)
       self.rz0 = tq.RZ(has_params=True, trainable=True)
       self.crx0 = tq.CRX(has_params=True, trainable=True)

     def forward(self, x):
       bsz = x.shape[0]
       # down-sample the image
       x = F.avg_pool2d(x, 6).view(bsz, 16)

       # create a quantum device to run the gates
       qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

       # encode the classical image to quantum domain
       for k, gate in enumerate(self.encoder_gates):
         gate(qdev, wires=k % self.n_wires, params=x[:, k])

       # add some trainable gates (need to instantiate ahead of time)
       self.rx0(qdev, wires=0)
       self.ry0(qdev, wires=1)
       self.rz0(qdev, wires=3)
       self.crx0(qdev, wires=[0, 2])

       # add some more non-parameterized gates (add on-the-fly)
       qdev.h(wires=3)
       qdev.sx(wires=2)
       qdev.cnot(wires=[3, 0])
       qdev.qubitunitary(wires=[1, 2], params=[[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 0, 1j],
                                               [0, 0, -1j, 0]])

       # perform measurement to get expectations (back to classical domain)
       x = self.measure(qdev).reshape(bsz, 2, 2)

       # classification
       x = x.sum(-1).squeeze()
       x = F.log_softmax(x, dim=1)

       return x

VQE Example
-----------

Train a quantum circuit to perform VQE task. Quito quantum computer as
in `simple_vqe.py <./examples/simple_vqe/simple_vqe.py>`__ script:

.. code:: python

   cd examples/simple_vqe
   python simple_vqe.py

MNIST Example
-------------

Train a quantum circuit to perform MNIST task and deploy on the real IBM
Quito quantum computer as in
`mnist_example.py <./examples/simple_mnist/mnist_example_no_binding.py>`__
script:

.. code:: python

   cd examples/simple_mnist
   python mnist_example.py

Files
-----

+-----------------------------------+-----------------------------------+
| File                              | Description                       |
+===================================+===================================+
| devices.py                        | QuantumDevice class which stores  |
|                                   | the statevector                   |
+-----------------------------------+-----------------------------------+
| encoding.py                       | Encoding layers to encode         |
|                                   | classical values to quantum       |
|                                   | domain                            |
+-----------------------------------+-----------------------------------+
| functional.py                     | Quantum gate functions            |
+-----------------------------------+-----------------------------------+
| operators.py                      | Quantum gate classes              |
+-----------------------------------+-----------------------------------+
| layers.py                         | Layer templates such as           |
|                                   | RandomLayer                       |
+-----------------------------------+-----------------------------------+
| measure.py                        | Measurement of quantum states to  |
|                                   | get classical values              |
+-----------------------------------+-----------------------------------+
| graph.py                          | Quantum gate graph used in static |
|                                   | mode                              |
+-----------------------------------+-----------------------------------+
| super_layer.py                    | Layer templates for SuperCircuits |
+-----------------------------------+-----------------------------------+
| plugins/qiskit\*                  | Convertors and processors for     |
|                                   | easy deployment on IBMQ           |
+-----------------------------------+-----------------------------------+
| examples/                         | More examples for training QML    |
|                                   | and VQE models                    |
+-----------------------------------+-----------------------------------+

Coding Style
------------

torchquantum uses pre-commit hooks to ensure Python style consistency
and prevent common mistakes in its codebase.

To enable it pre-commit hooks please reproduce:

.. code:: bash

   pip install pre-commit
   pre-commit install

Papers using TorchQuantum
-------------------------

-  [HPCA’22] `Wang et al., “QuantumNAS: Noise-Adaptive Search for Robust
   Quantum Circuits” <https://arxiv.org/abs/2107.10845>`__
-  [DAC’22] `Wang et al., “QuantumNAT: Quantum Noise-Aware Training with
   Noise Injection, Quantization and
   Normalization” <https://arxiv.org/abs/2110.11331>`__
-  [DAC’22] `Wang et al., “QOC: Quantum On-Chip Training with Parameter
   Shift and Gradient Pruning” <https://arxiv.org/abs/2202.13239>`__
-  [QCE’22] `Liang et al., “Variational Quantum Pulse
   Learning” <https://arxiv.org/abs/2203.17267>`__
-  [ICCAD’22] `Hu et al., “Quantum Neural Network
   Compression” <https://arxiv.org/abs/2207.01578>`__
-  [ICCAD’22] `Wang et al., “QuEst: Graph Transformer for Quantum
   Circuit Reliability Estimation” <https://arxiv.org/abs/2210.16724>`__
-  [ICML Workshop] `Yun et al., “Slimmable Quantum Federated
   Learning” <https://dynn-icml2022.github.io/spapers/paper_7.pdf>`__
-  [IEEE ICDCS] `Yun et al., “Quantum Multi-Agent Reinforcement Learning
   via Variational Quantum Circuit
   Design” <https://ieeexplore.ieee.org/document/9912289>`__

   .. raw:: html

      <details>

   .. raw:: html

      <summary>

   Manuscripts

   .. raw:: html

      </summary>

   .. rubric:: Manuscripts
      :name: manuscripts

   -  `Yun et al., “Projection Valued Measure-based Quantum Machine
      Learning for Multi-Class
      Classification” <https://arxiv.org/abs/2210.16731>`__
   -  `Baek et al., “3D Scalable Quantum Convolutional Neural Networks
      for Point Cloud Data Processing in Classification
      Applications” <https://arxiv.org/abs/2210.09728>`__
   -  `Baek et al., “Scalable Quantum Convolutional Neural
      Networks” <https://arxiv.org/abs/2209.12372>`__
   -  `Yun et al., “Quantum Multi-Agent Meta Reinforcement
      Learning” <https://arxiv.org/abs/2208.11510>`__

.. raw:: html

   </details>

Dependencies
------------

-  3.9 >= Python >= 3.7 (Python 3.10 may have the ``concurrent`` package
   issue for Qiskit)
-  PyTorch >= 1.8.0
-  configargparse >= 0.14
-  GPU model training requires NVIDIA GPUs

Contact
-------

TorchQuantum `Forum <https://qmlsys.hanruiwang.me>`__

Hanrui Wang hanrui@mit.edu

Contributors
------------

Jiannan Cao, Jessica Ding, Jiai Gu, Song Han, Zhirui Hu, Zirui Li,
Zhiding Liang, Pengyu Liu, Yilian Liu, Mohammadreza Tavasoli, Hanrui
Wang, Zhepeng Wang, Zhuoyang Ye

Citation
--------

::

   @inproceedings{hanruiwang2022quantumnas,
       title     = {Quantumnas: Noise-adaptive search for robust quantum circuits},
       author    = {Wang, Hanrui and Ding, Yongshan and Gu, Jiaqi and Li, Zirui and Lin, Yujun and Pan, David Z and Chong, Frederic T and Han, Song},
       booktitle = {The 28th IEEE International Symposium on High-Performance Computer Architecture (HPCA-28)},
       year      = {2022}
   }
