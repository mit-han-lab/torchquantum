# torchquantum
A PyTorch-centric hybrid classical-quantum dynamic neural networks framework.

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/Hanrui-Wang/pytorch-quantum/blob/master/LICENSE)

## News
- v0.1.0 coming soon.

## Features
- Support batch mode inference and training on CPU/GPU.
- Support dynamic graph.
- Support easy deployment on real quantum devices such as IBMQ.

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
- GPU model training requires NVIDIA GPUs and NCCL

## Run
MNIST training with a hybrid classical and quantum network.

```python
python examples/train.py examples/configs/mnist/train/t_hybrid.yml
```
