.. raw:: html

   <p align="center">

.. raw:: html

   </p>

::


Welcome to TorchQuantum's documentation!
============


   @inproceedings{hanruiwang2022quantumnas,
       title     = {Quantumnas: Noise-adaptive search for robust quantum circuits},
       author    = {Wang, Hanrui and Ding, Yongshan and Gu, Jiaqi and Lin, Yujun and Pan, David Z and Chong, Frederic T and Han, Song},
       booktitle = {The 28th IEEE International Symposium on High-Performance Computer Architecture (HPCA-28)},
       year      = {2022}
   }



A PyTorch-based hybrid classical-quantum dynamic neural networks
framework.

|MIT License|

News
----

-  Colab examples are available in the `artifact <./artifact>`__ folder.
-  Our recent paper `“QuantumNAS: Noise-Adaptive Search for Robust
   Quantum Circuits” <https://arxiv.org/abs/2107.10845>`__ is accepted
   to HPCA 2022. We are working on updating the repo to add more
   examples soon!
-  Add a simple `example script <./mnist_example.py>`__ using quantum
   gates to do MNIST classification.
-  v0.0.1 available. Feedbacks are highly welcomed!

Installation
------------

.. code:: bash

   git clone https://github.com/mit-han-lab/torchquantum.git
   cd torchquantum
   pip install --editable .

.. raw:: html

   <!-- ## Brief Intro Video
   [![Watch the video](https://hanlab.mit.edu/projects/qmlsys/assets/torchquantum_intro.png)](https://qmlsys.mit.edu/assets/torchquantum_intro.mp4)

    -->

Usage
-----

Construct quantum NN models as simple as constructing a normal pytorch
model.

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

Features
--------

-  Easy construction of parameterized quantum circuits in PyTorch.
-  Support **batch mode** inference and training on CPU/GPU.
-  Support **dynamic computation graph** for easy debugging.
-  Support easy **deployment on real quantum devices** such as IBMQ.

TODOs
-----

-  ☒ Support more gates
-  ☒ Support compile a unitary with descriptions to speedup training
-  ☐ Support other measurements other than analytic method
-  ☒ In einsum support multiple qubit sharing one letter. So that more
   than 26 qubit can be simulated.
-  ☒ Support bmm based implementation to solve scalability issue
-  ☒ Support conversion from torchquantum to qiskit

Dependencies
------------

-  Python >= 3.7
-  PyTorch >= 1.8.0
-  configargparse >= 0.14
-  GPU model training requires NVIDIA GPUs

MNIST Example
-------------

Train a quantum circuit to perform MNIST task and deploy on the real IBM
Yorktown quantum computer as in
`mnist_example.py <./mnist_example.py>`__ script:

.. code:: python

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

More Examples
-------------

The ``examples/`` folder contains more examples to train the QML and VQE
models. Example usage for a QML circuit:

.. code:: python

   # train the circuit with 36 params in the U3+CU3 space
   python examples/train.py examples/configs/mnist/four0123/train/baseline/u3cu3_s0/rand/param36.yml

   # evaluate the circuit with torchquantum
   python examples/eval.py examples/configs/mnist/four0123/eval/tq/all.yml --run-dir=runs/mnist.four0123.train.baseline.u3cu3_s0.rand.param36

   # evaluate the circuit with real IBMQ-Yorktown quantum computer
   python examples/eval.py examples/configs/mnist/four0123/eval/x2/real/opt2/300.yml --run-dir=runs/mnist.four0123.train.baseline.u3cu3_s0.rand.param36

Example usage for a VQE circuit:

.. code:: python

   # Train the VQE circuit for h2
   python examples/train.py examples/configs/vqe/h2/train/baseline/u3cu3_s0/human/param12.yml

   # evaluate the VQE circuit with torchquantum
   python examples/eval.py examples/configs/vqe/h2/eval/tq/all.yml --run-dir=runs/vqe.h2.train.baseline.u3cu3_s0.human.param12/

   # evaluate the VQE circuit with real IBMQ-Yorktown quantum computer
   python examples/eval.py examples/configs/vqe/h2/eval/x2/real/opt2/all.yml --run-dir=runs/vqe.h2.train.baseline.u3cu3_s0.human.param12/

Detailed documentations coming soon.

QuantumNAS
----------

Quantum noise is the key challenge in Noisy Intermediate-Scale Quantum
(NISQ) computers. Previous work for mitigating noise has primarily
focused on gate-level or pulse-level noise-adaptive compilation.
However, limited research efforts have explored a higher level of
optimization by making the quantum circuits themselves resilient to
noise. We propose QuantumNAS, a comprehensive framework for
noise-adaptive co-search of the variational circuit and qubit mapping.
Variational quantum circuits are a promising approach for constructing
QML and quantum simulation. However, finding the best variational
circuit and its optimal parameters is challenging due to the large
design space and parameter training cost. We propose to decouple the
circuit search and parameter training by introducing a novel
SuperCircuit. The SuperCircuit is constructed with multiple layers of
pre-defined parameterized gates and trained by iteratively sampling and
updating the parameter subsets (SubCircuits) of it. It provides an
accurate estimation of SubCircuits performance trained from scratch.
Then we perform an evolutionary co-search of SubCircuit and its qubit
mapping. The SubCircuit performance is estimated with parameters
inherited from SuperCircuit and simulated with real device noise models.
Finally, we perform iterative gate pruning and finetuning to remove
redundant gates. Extensively evaluated with 12 QML and VQE benchmarks on
10 quantum comput, QuantumNAS significantly outperforms baselines. For
QML, QuantumNAS is the first to demonstrate over 95% 2-class, 85%
4-class, and 32% 10-class classification accuracy on real QC. It also
achieves the lowest eigenvalue for VQE tasks on H2, H2O, LiH, CH4, BeH2
compared with UCCSD. We also open-source torchquantum for fast training
of parameterized quantum circuits to facilitate future research.

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

QuantumNAS Framework overview:

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

QuantumNAS models achieve higher robustness and accuracy than other
baseline models:

.. raw:: html

   <p align="center">

.. raw:: html

   </p>

Contact
-------

Hanrui Wang (hanrui@mit.edu)

.. |MIT License| image:: https://img.shields.io/apm/l/atomic-design-ui.svg?
   :target: https://github.com/mit-han-lab/torchquantum/blob/master/LICENSE
