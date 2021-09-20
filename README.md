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
For MNIST-36 dataset
```bash
python3 examples/train.py examples/configs/mnist_front500/two36/4qubits/train/noaddnoise/nonorm/seth_0/n1b1/ibmq_manila/realqcTrainRealqcValid.yml --gpu=4
```
For MNIST-0123 dataset
```bash
python3 examples/train.py examples/configs/mnist_front500/four0123/4qubits/train/noaddnoise/nonorm/seth_0/n1b3/ibmq_manila/realqcTrainRealqcValid.yml --gpu=5
```
Choose the training mode: clsTrainClsValid.yml/clsTrainRealqcValid.yml/realqcTrainRealqcValid.yml