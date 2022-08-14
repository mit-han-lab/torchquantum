.. toctree::
   :maxdepth: 1
   :caption: API

   api_torchquantum
   api_functional
   api_operators

.. toctree::
   :maxdepth: 1
   :caption: Usage

   usage_installation
   examples/index.rst
   
.. raw:: html

   <p align="center">

.. raw:: html

   </p>

.. raw:: html

   <h2>

.. raw:: html

   <p align="center">

A PyTorch Library for Quantum Simulation and Quantum Machine Learning

.. raw:: html

   </p>

.. raw:: html

   </h2>

.. raw:: html

   <h3>

.. raw:: html

   <p align="center">

Faster, Scalable, Easy Debugging, Easy Deployment on Real Machine

.. raw:: html

   </p>

.. raw:: html

   </h3>

|MIT License| |Read the Docs| |Discourse status| |Website|

👋 Welcome
=========

- What it is doing
Quantum simulation framework based on PyTorch. It supports statevector
simulation and pulse simulation (coming soon) on GPUs. It can scale up
to the simulation of 25+ qubits with multiple GPUs. 

- Who will benefit
Researchers on quantum algorithm design, parameterized quantum
circuit training, quantum optimal control, quantum machine learning,
quantum neural networks. 

- Differences from Qiskit/Pennylane 
Dynamic computatioh graph, automatic gradient computation, fast GPU support,
batch model tersorized processing.

Features
--------

-  Easy construction and simulation of quantum circuits in **PyTorch**
-  **Dynamic computation graph** for easy debugging
-  **Gradient support** via autograd
-  **Batch mode** inference and training on **CPU/GPU**.
-  Easy **deployment on real quantum devices** such as IBMQ
-  **Easy hybrid classical-quantum** model construction
-  (coming soon) **pulse-level simulation**

News
----

-  Welcome to contribute! Please contact us or post in the
   `forum <https://qmlsys.hanruiwang.me>`__ if you want to have new
   examples implemented by TorchQuantum or any other questions.
-  Qmlsys website goes online:
   `qmlsys.mit.edu <https://qmlsys.mit.edu>`__

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

   x = tq.QuantumDevice(n_wires=1)

   tqf.hadamard(x, wires=0)
   tqf.x(x, wires=1)
   tqf.cnot(x, wires=[0, 1])

   # print the current state (dynamic computation graph supported)
   print(x.states)

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
`VQA for State Prepration <examples/train_state_prep>`__.

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
       self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
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

       # reset qubit states
       self.q_device.reset_states(bsz)

       # encode the classical image to quantum domain
       for k, gate in enumerate(self.encoder_gates):
         gate(self.q_device, wires=k % self.n_wires, params=x[:, k])

       # add some trainable gates (need to instantiate ahead of time)
       self.rx0(self.q_device, wires=0)
       self.ry0(self.q_device, wires=1)
       self.rz0(self.q_device, wires=3)
       self.crx0(self.q_device, wires=[0, 2])

       # add some more non-parameterized gates (add on-the-fly)
       tqf.hadamard(self.q_device, wires=3)
       tqf.sx(self.q_device, wires=2)
       tqf.cnot(self.q_device, wires=[3, 0])
       tqf.qubitunitary(self.q_device0, wires=[1, 2], params=[[1, 0, 0, 0],
                                                              [0, 1, 0, 0],
                                                              [0, 0, 0, 1j],
                                                              [0, 0, -1j, 0]])

       # perform measurement to get expectations (back to classical domain)
       x = self.measure(self.q_device).reshape(bsz, 2, 2)

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

+------------------+--------------------------------------------------+
| File             | Description                                      |
+==================+==================================================+
| devices.py       | QuantumDevice class which stores the statevector |
+------------------+--------------------------------------------------+
| encoding.py      | Encoding layers to encode classical values to    |
|                  | quantum domain                                   |
+------------------+--------------------------------------------------+
| functional.py    | Quantum gate functions                           |
+------------------+--------------------------------------------------+
| operators.py     | Quantum gate classes                             |
+------------------+--------------------------------------------------+
| layers.py        | Layer templates such as RandomLayer              |
+------------------+--------------------------------------------------+
| measure.py       | Measurement of quantum states to get classical   |
|                  | values                                           |
+------------------+--------------------------------------------------+
| graph.py         | Quantum gate graph used in static mode           |
+------------------+--------------------------------------------------+
| super_layer.py   | Layer templates for SuperCircuits                |
+------------------+--------------------------------------------------+
| plugins/qiskit\* | Convertors and processors for easy deployment on |
|                  | IBMQ                                             |
+------------------+--------------------------------------------------+
| examples/        | More examples for training QML and VQE models    |
+------------------+--------------------------------------------------+

Papers using TorchQuantum
-------------------------

-  [HPCA’22] `QuantumNAS: Noise-Adaptive Search for Robust Quantum
   Circuits <artifact>`__
-  [DAC’22] `QuantumNAT: Quantum Noise-Aware Training with Noise
   Injection, Quantization and
   Normalization <https://arxiv.org/abs/2110.11331>`__
-  [DAC’22] `QOC: Quantum On-Chip Training with Parameter Shift and
   Gradient Pruning <https://arxiv.org/abs/2202.13239>`__
-  [QCE’22] `Variational Quantum Pulse
   Learning <https://arxiv.org/abs/2203.17267>`__

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

Citation
--------

::

   @inproceedings{hanruiwang2022quantumnas,
       title     = {Quantumnas: Noise-adaptive search for robust quantum circuits},
       author    = {Wang, Hanrui and Ding, Yongshan and Gu, Jiaqi and Li, Zirui and Lin, Yujun and Pan, David Z and Chong, Frederic T and Han, Song},
       booktitle = {The 28th IEEE International Symposium on High-Performance Computer Architecture (HPCA-28)},
       year      = {2022}
   }

.. |MIT License| image:: https://img.shields.io/apm/l/atomic-design-ui.svg?
   :target: https://github.com/mit-han-lab/torchquantum/blob/master/LICENSE
.. |Read the Docs| image:: https://img.shields.io/readthedocs/torchquantum
   :target: https://torchquantum-doc.readthedocs.io/
.. |Discourse status| image:: https://img.shields.io/discourse/status?server=https%3A%2F%2Fqmlsys.hanruiwang.me%2F
   :target: https://qmlsys.hanruiwang.me
.. |Website| image:: https://img.shields.io/website?up_message=qmlsys&url=https%3A%2F%2Fqmlsys.mit.edu
   :target: https://qmlsys.mit.edu
