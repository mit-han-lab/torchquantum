# pytorch-quantum
A PyTorch-centric hybrid classical-quantum dynamic neural networks framework.

## News
- v0.0.1 coming soon...

## Features
- Support batch mode inference and training on CPU/GPU.
- Support dynamic graph.
- Support easy deployment on real quantum devices such as IBMQ.

## TODOs
- [ ] Support more gates
- [ ] Support compile a unitary with descriptions to speedup training
- [ ] Support other measurements other than analytic method

## Dependencies
- Python >= 3.7
- PyTorch >= 1.8 (**WARNING** Pytorch supports complex value matmul starting v1.8 so we have to use unstable v1.8 nightly.)
- configargparse >= 0.14
- GPU model training requires NVIDIA GPUs and NCCL

## Run
MNIST training with a hybrid classical and quantum network.
```python
python examples/mnist/test0.py
```
