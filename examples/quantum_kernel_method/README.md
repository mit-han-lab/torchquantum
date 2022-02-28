## Quantum Kernel Method

Authors: Zirui Li, Hanrui Wang

Use Colab to run this example: [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mit-han-lab/torchquantum/blob/master/examples/quantum_kernel_method/quantum_kernel_method.ipynb)

See this tutorial video for detailed explanations:

[![video](https://img.youtube.com/vi/5sfF4TRxFro/0.jpg)](https://www.youtube.com/watch?v=5sfF4TRxFro)

Referece: [Supervised quantum machine learning models are kernel methods
](https://arxiv.org/abs/2101.11020)


### Outline
1. Introduction to Quantum Kernel Methods.
2. Build and train an SVM using Quantum Kernel Methods.

In this tutorial, we use `tq.op_name_dict`, `tq.functional.func_name_dict` and `tq.QuantumDevice` from TorchQuantum.

You can learn how to build a Quantum kernel function and train an SVM with the quantum kernel from this tutorial.


[comment]: <> (#%% md)

## Introduction to Quantum Kernel Methods.


[comment]: <> (#%% md)

### Kernel Methods
Kernels or kernel methods (also called Kernel functions) are sets of different types of algorithms that are being used for pattern analysis. They are used to solve a non-linear problem by a linear classifier. Kernels Methods are employed in SVM (Support Vector Machines) which are often used in classification and regression problems. The SVM uses what is called a “Kernel Trick” where the data is transformed and an optimal boundary is found for the possible outputs.


#### Quantum Kernel
Quantum circuit can transfer the data to a high dimension Hilbert space which is hard to simulate on classical computer. Using kernel methods based on this Hilbert space can achieve unexpected performance.

[comment]: <> (#%% md)

### How to evaluate the distance in Hilbert space?
Assume S(x) is the unitary that transfer data x to the state in Hilbert space. To evaluate the inner product between S(x) and S(y), we add a Transpose Conjugation of S(y) behind S(x) and measure the probability that the state falls on $|00\cdots0\rangle$

[comment]: <> (#%% md)


<div align="center">
<img src="https://github.com/mit-han-lab/torchquantum/blob/master/figs/kernel.png?raw=true" alt="conv-full-layer" width="600">
</div>