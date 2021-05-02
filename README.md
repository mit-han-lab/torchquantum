# torchquantum
A PyTorch-centric hybrid classical-quantum dynamic neural networks framework.

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/Hanrui-Wang/pytorch-quantum/blob/master/LICENSE)

## News
- Add a simple [example script](./mnist_example.py) using quantum gates to do MNIST 
  classification.
- v0.0.1 available. Feedbacks are highly welcomed!

## Installation
```bash
git clone https://github.com/Hanrui-Wang/pytorch-quantum.git
cd pytorch-quantum
pip install --editable .
```

## Usage
Construct quantum NN models as simple as constructing a normal pytorch model.
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

## Features
- Easy construction of parameterized quantum circuits in PyTorch.
- Support **batch mode** inference and training on CPU/GPU.
- Support **dynamic computation graph** for easy debugging.
- Support easy **deployment on real quantum devices** such as IBMQ.

## TODOs
- [x] Support more gates
- [x] Support compile a unitary with descriptions to speedup training
- [ ] Support other measurements other than analytic method
- [x] In einsum support multiple qubit sharing one letter. So that more 
  than 26 qubit can be simulated.
- [x] Support bmm based implementation to solve 
  scalability issue
- [x] Support conversion from torchquantum to qiskit

## Dependencies
- Python >= 3.7
- PyTorch >= 1.8.0 
- configargparse >= 0.14
- GPU model training requires NVIDIA GPUs

## MNIST Example
Train a quantum circuit to perform MNIST task and deploy on the real IBM
Yorktown quantum computer as in [mnist_example.py](./mnist_example.py)
script:
```python
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

## More Examples
The `examples/` folder contains more examples to train the QML and VQE
models. Example usage for a QML circuit:
```python
# train the circuit with 36 params in the U3+CU3 space
python examples/train.py examples/configs/mnist/four0123/train/baseline/u3cu3_s0/rand/param36.yml

# evaluate the circuit with torchquantum
python examples/eval.py examples/configs/mnist/four0123/eval/tq/all.yml --run-dir=runs/mnist.four0123.train.baseline.u3cu3_s0.rand.param36

# evaluate the circuit with real IBMQ-Yorktown quantum computer
python examples/eval.py examples/configs/mnist/four0123/eval/x2/real/opt2/300.yml --run-dir=runs/mnist.four0123.train.baseline.u3cu3_s0.rand.param36
```

Example usage for a VQE circuit:
```python
# Train the VQE circuit for h2
python examples/train.py examples/configs/vqe/h2/train/baseline/u3cu3_s0/human/param12.yml

# evaluate the VQE circuit with torchquantum
python examples/eval.py examples/configs/vqe/h2/eval/tq/all.yml --run-dir=runs/vqe.h2.train.baseline.u3cu3_s0.human.param12/

# evaluate the VQE circuit with real IBMQ-Yorktown quantum computer
python examples/eval.py examples/configs/vqe/h2/eval/x2/real/opt2/all.yml --run-dir=runs/vqe.h2.train.baseline.u3cu3_s0.human.param12/

```

Detailed documentations coming soon.

## Contact
Hanrui Wang (hanrui@mit.edu)
