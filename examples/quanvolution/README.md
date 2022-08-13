# Quantum Convolution
## Quantum Convolution (Quanvolution) for MNIST image classification

Authors: Zirui Li, Hanrui Wang

se Colab to run this example: [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mit-han-lab/torchquantum/blob/master/examples/quanvolution/quanvolution.ipynb)

See this tutorial video for detailed explanations: 

[![video](https://img.youtube.com/vi/-Grfxkg3-DI/0.jpg)](https://www.youtube.com/watch?v=-Grfxkg3-DI)

Referece: [Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits](https://arxiv.org/abs/1904.04767)

### Outline
1. Introduction to Quanvolutional Neural Network.
2. Build and train a Quanvolutional Neural Network.
- a.  Compare Quanvolutional Neural Network with a classic model.
- b. Evaluate on real quantum computer.
3. Compare multiple models with or without a trainable quanvolutional filter.

[comment]: <> (#%% md)

In this tutorial, we use `tq.QuantumDevice`, `tq.GeneralEncoder`, `tq.RandomLayer`, `tq.MeasureAll`, `tq.PauliZ` class from TrochQuantum.

You can learn how to build, train and evaluate a quanvolutional filter using TorchQuantum in this tutorial.

[comment]: <> (#%% md)

## Introduction to Quanvolutional Neural Network.
### Convolutional Neural Network
Convolutional neural network is a classic neural network genre, mostly applied to anylize visual images. They are known for their convolutional layers that perform convolution. Typically the convolution operation is the Frobenius inner product of the convolution filter with the input image followed by an activation function. The convolution filter slides along the input image and generates a feature map. We can use the feature map for classification.

<div align="center">
<img src="https://github.com/mit-han-lab/torchquantum/blob/master/figs/conv-full-layer.gif?raw=true" alt="conv-full-layer" width="300">
</div>

### Quantum convolution
One can extend the same idea also to the context of quantum variational circuits. Replace the classical convolution filters with variational quantum circuits and we get quanvolutional neural networks with quanvolutional filters. The quanvolutional filters perform more complex operations in a higher dimension Hilbert space than Frobenius inner product. Therefore, quanvolutional filters have more potential than traditional convolution filters.

<div align="center">
<img src="https://github.com/mit-han-lab/torchquantum/blob/master/figs/hybridmodel.png?raw=true" alt="conv-full-layer" width="800">
</div>
