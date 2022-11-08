<p align="center">
<img src="https://github.com/mit-han-lab/pytorch-quantum/blob/master/docs/torchquantum_logo.jpg" alt="torchquantum Logo" width="450">
</p>

<h2><p align="center">A PyTorch Library for Quantum Simulation and Quantum Machine Learning</p></h2>
<h3><p align="center">Faster, Scalable, Easy Debugging, Easy Deployment on Real Machine</p></h3>


<p align="center">
    <a href="https://github.com/mit-han-lab/torchquantum/blob/master/LICENSE">
        <img alt="MIT License" src="https://img.shields.io/apm/l/atomic-design-ui.svg?">
    </a>
    <a href="https://torchquantum.readthedocs.io/">
        <img alt="Documentation" src="https://img.shields.io/readthedocs/torchquantum/master">
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





# 👋 Welcome

#### What it is doing
Quantum simulation framework based on PyTorch. It supports statevector simulation and pulse simulation (coming soon) on GPUs. It can scale up to the simulation of 30+ qubits with multiple GPUs.
#### Who will benefit
Researchers on quantum algorithm design, parameterized quantum circuit training, quantum optimal control, quantum machine learning, quantum neural networks.
#### Differences from Qiskit/Pennylane
Dynamic computation graph, automatic gradient computation, fast GPU support, batch model tersorized processing.

## News
- Added support for controlled unitary
- v0.1.2 Available!
- Join our [Slack](https://join.slack.com/t/torchquantum/shared_invite/zt-1d830lvph-E~6IlXdtdQLoGLljLp_xFg) for real time support! 
- Welcome to contribute! Please contact us or post in the [forum](https://qmlsys.hanruiwang.me) if you want to have new examples implemented by TorchQuantum or any other questions.
- Qmlsys website goes online: [qmlsys.mit.edu](https://qmlsys.mit.edu)

## Features
- Easy construction and simulation of quantum circuits in **PyTorch**
- **Dynamic computation graph** for easy debugging
- **Gradient support** via autograd
- **Batch mode** inference and training on **CPU/GPU**.
- Easy **deployment on real quantum devices** such as IBMQ
- **Easy hybrid classical-quantum** model construction
- (coming soon) **pulse-level simulation**



## Installation
```bash
pip install torchquantum

```
## Basic Usage 1

```python
import torchquantum as tq
import torchquantum.functional as tqf

state = tq.QuantumState(n_wires=2)

state.h(wires=0)
state.cnot(wires=[0, 1])
tqf.h(state, wires=1)
tqf.x(state, wires=1)

# print the current state (dynamic computation graph supported)
print(state)
print(tq.measure(state, n_shots=1024))

```

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
```



## Guide to the examples
We also prepare many example and tutorials using TorchQuantum.

For **beginning level**, you may check [QNN for MNIST](examples/simple_mnist), [Quantum Convolution (Quanvolution)](examples/quanvolution) and [Quantum Kernel Method](examples/quantum_kernel_method), and [Quantum Regression](examples/regression).

For **intermediate level**, you may check [Amplitude Encoding for MNIST](examples/amplitude_encoding_mnist), [Clifford gate QNN](examples/clifford_qnn), [Save and Load QNN models](examples/save_load_example), [PauliSum Operation](examples/PauliSumOp), [How to convert tq to Qiskit](examples/converter_tq_qiskit).

For **expert**, you may check [Parameter Shift on-chip Training](examples/param_shift_onchip_training), [VQA Gradient Pruning](examples/gradient_pruning), [VQE](examples/simple_vqe),  [VQA for State Prepration](examples/train_state_prep).


## Usage
Construct parameterized quantum circuit models as simple as constructing a normal pytorch model.
```python
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

```

## VQE Example
Train a quantum circuit to perform VQE task.
Quito quantum computer as in [simple_vqe.py](./examples/simple_vqe/simple_vqe.py)
script:
```python
cd examples/simple_vqe
python simple_vqe.py
```


## MNIST Example
Train a quantum circuit to perform MNIST task and deploy on the real IBM
Quito quantum computer as in [mnist_example.py](./examples/simple_mnist/mnist_example_no_binding.py)
script:
```python
cd examples/simple_mnist
python mnist_example.py
```

## Files
| File      | Description |
| ----------- | ----------- |
| devices.py      | QuantumDevice class which stores the statevector |
| encoding.py   | Encoding layers to encode classical values to quantum domain |
| functional.py   | Quantum gate functions |
| operators.py   | Quantum gate classes |
| layers.py   | Layer templates such as RandomLayer |
| measure.py   | Measurement of quantum states to get classical values |
| graph.py   | Quantum gate graph used in static mode |
| super_layer.py   | Layer templates for SuperCircuits |
| plugins/qiskit*   | Convertors and processors for easy deployment on IBMQ |
| examples/| More examples for training QML and VQE models |

[comment]: <> (## More Examples)

[comment]: <> (The `examples/` folder contains more examples to train the QML and VQE)

[comment]: <> (models. Example usage for a QML circuit:)

[comment]: <> (```python)

[comment]: <> (# train the circuit with 36 params in the U3+CU3 space)

[comment]: <> (python examples/train.py examples/configs/mnist/four0123/train/baseline/u3cu3_s0/rand/param36.yml)

[comment]: <> (# evaluate the circuit with torchquantum)

[comment]: <> (python examples/eval.py examples/configs/mnist/four0123/eval/tq/all.yml --run-dir=runs/mnist.four0123.train.baseline.u3cu3_s0.rand.param36)

[comment]: <> (# evaluate the circuit with real IBMQ-Yorktown quantum computer)

[comment]: <> (python examples/eval.py examples/configs/mnist/four0123/eval/x2/real/opt2/300.yml --run-dir=runs/mnist.four0123.train.baseline.u3cu3_s0.rand.param36)

[comment]: <> (```)

[comment]: <> (Example usage for a VQE circuit:)

[comment]: <> (```python)

[comment]: <> (# Train the VQE circuit for h2)

[comment]: <> (python examples/train.py examples/configs/vqe/h2/train/baseline/u3cu3_s0/human/param12.yml)

[comment]: <> (# evaluate the VQE circuit with torchquantum)

[comment]: <> (python examples/eval.py examples/configs/vqe/h2/eval/tq/all.yml --run-dir=runs/vqe.h2.train.baseline.u3cu3_s0.human.param12/)

[comment]: <> (# evaluate the VQE circuit with real IBMQ-Yorktown quantum computer)

[comment]: <> (python examples/eval.py examples/configs/vqe/h2/eval/x2/real/opt2/all.yml --run-dir=runs/vqe.h2.train.baseline.u3cu3_s0.human.param12/)

[comment]: <> (```)

[comment]: <> (Detailed documentations coming soon.)

[comment]: <> (## QuantumNAS)

[comment]: <> (Quantum noise is the key challenge in Noisy Intermediate-Scale Quantum &#40;NISQ&#41; computers. Previous work for mitigating noise has primarily focused on gate-level or pulse-level noise-adaptive compilation. However, limited research efforts have explored a higher level of optimization by making the quantum circuits themselves resilient to noise. We propose QuantumNAS, a comprehensive framework for noise-adaptive co-search of the variational circuit and qubit mapping. Variational quantum circuits are a promising approach for constructing QML and quantum simulation. However, finding the best variational circuit and its optimal parameters is challenging due to the large design space and parameter training cost. We propose to decouple the circuit search and parameter training by introducing a novel SuperCircuit. The SuperCircuit is constructed with multiple layers of pre-defined parameterized gates and trained by iteratively sampling and updating the parameter subsets &#40;SubCircuits&#41; of it. It provides an accurate estimation of SubCircuits performance trained from scratch. Then we perform an evolutionary co-search of SubCircuit and its qubit mapping. The SubCircuit performance is estimated with parameters inherited from SuperCircuit and simulated with real device noise models. Finally, we perform iterative gate pruning and finetuning to remove redundant gates. Extensively evaluated with 12 QML and VQE benchmarks on 10 quantum comput, QuantumNAS significantly outperforms baselines. For QML, QuantumNAS is the first to demonstrate over 95% 2-class, 85% 4-class, and 32% 10-class classification accuracy on real QC. It also achieves the lowest eigenvalue for VQE tasks on H2, H2O, LiH, CH4, BeH2 compared with UCCSD. We also open-source torchquantum for fast training of parameterized quantum circuits to facilitate future research.)

[comment]: <> (<p align="center">)

[comment]: <> (<img src="https://hanruiwang.me/project_pages/quantumnas/assets/teaser.jpg" alt="torchquantum teaser" width="550">)

[comment]: <> (</p>)

[comment]: <> (QuantumNAS Framework overview:)

[comment]: <> (<p align="center">)

[comment]: <> (<img src="https://hanruiwang.me/project_pages/quantumnas/assets/overview.jpg" alt="torchquantum overview" width="1000">)

[comment]: <> (</p>)

[comment]: <> (QuantumNAS models achieve higher robustness and accuracy than other baseline models:)

[comment]: <> (<p align="center">)

[comment]: <> (<img src="https://hanruiwang.me/project_pages/quantumnas/assets/results.jpg" alt="torchquantum results" width="550">)

[comment]: <> (</p>)

## Papers using TorchQuantum
- [HPCA'22] [QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits](artifact)
- [DAC'22] [QuantumNAT: Quantum Noise-Aware Training with Noise Injection, Quantization and Normalization](https://arxiv.org/abs/2110.11331)
- [DAC'22] [QOC: Quantum On-Chip Training with Parameter Shift and Gradient Pruning](https://arxiv.org/abs/2202.13239)
- [QCE'22] [Variational Quantum Pulse Learning](https://arxiv.org/abs/2203.17267)
- [ICCAD'22] [Quantum Neural Network Compression](https://arxiv.org/abs/2207.01578)
- [ICCAD'22] [Graph Transformer for Quantum Circuit Reliability Prediction]()


## Dependencies
- 3.9 >= Python >= 3.7 (Python 3.10 may have the `concurrent` package issue for Qiskit)
- PyTorch >= 1.8.0 
- configargparse >= 0.14
- GPU model training requires NVIDIA GPUs

## Contact
TorchQuantum [Forum](https://qmlsys.hanruiwang.me)

Hanrui Wang [hanrui@mit.edu](mailto:hanrui@mit.edu)

## Contributors
Hanrui Wang, Jiannan Cao, Jessica Ding, Jiai Gu, Song Han, Zirui Li, Zhiding Liang, Pengyu Liu, Yilian Liu, Mohammadreza Tavasoli

## Citation
```
@inproceedings{hanruiwang2022quantumnas,
    title     = {Quantumnas: Noise-adaptive search for robust quantum circuits},
    author    = {Wang, Hanrui and Ding, Yongshan and Gu, Jiaqi and Li, Zirui and Lin, Yujun and Pan, David Z and Chong, Frederic T and Han, Song},
    booktitle = {The 28th IEEE International Symposium on High-Performance Computer Architecture (HPCA-28)},
    year      = {2022}
}
```

