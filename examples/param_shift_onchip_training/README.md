## On-chip Training of Quantum Neural Networks with parameter shift
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


Authors: Zirui Li, Hanrui Wang

Use Colab to run this example: [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mit-han-lab/torchquantum/blob/master/examples/param_shift_onchip_training/param_shift_onchip_training.ipynb)


[comment]: <> (#%% md)

### Outline
1. Introduction to Parameters Shift Rules.
2. Build the model and define the function.
3. Train the model.



[comment]: <> (#%% md)

## Introduction to Parameters Shift Rules

In this tutorial, you can learn parameters shift rules and how to use parameters shift rules to calculate gradients and use the gradient to train the model.

### Back Propagation

Previously, our quantum model was based on qiskit and pytorch. Once we did an inference of the model, pytorch will automatically build a computaion graph. We can calculate the gradients of each node in the computational graph in a reversed order based on the chain rule. This is called back propagation.
<div align="center">
<img src="https://github.com/mit-han-lab/torchquantum/blob/master/figs/bp.png?raw=true" alt="conv-full-layer" width="300">
</div>

### Parameters Shift Rules

As we all know, when executing a quantum circuit on real quantum machine, we can not observe the quantum state, so we can not use back propagation to calculate gradients when our circuits run on real quantum machine. Parameters shift rules offer us a technique to calculate gradients only by doing inference. For a circuit function $f(\theta)$, we can calculate $f'(\theta)$ by shifting $\theta$ twice and minus one result by the other and multiply with a factor. The figure below describes the workflow of how to calculate the gradient of a parameter in a 4-qubit circuit.

<div align="center">
<img src="https://github.com/mit-han-lab/torchquantum/blob/master/figs/ps.png?raw=true" alt="conv-full-layer" width="600">
</div>

Suppose an $m$-qubit quantum circuit is parametrized by $n$ parameters $\theta=[\theta_1,\cdots,\theta_i,\cdots,\theta_n]$, the expectation value of measurements of this circuit can be represented by a circuit function,
$$f(\theta)=\langle\psi|U(\theta_i)^{\dagger}\widehat{Q}U(\theta_i)|\psi\rangle, \quad f(\theta)\in\mathbb{R}^{m}, \theta\in\mathbb{R}^n.
$$
where $\theta_i$ is the scalar parameter whose gradient is to be calculated, and $U(\theta_i)$ is the gate where $\theta_i$ lies in.

Here, for notation simplicity, we have already absorbed the unitaries before $U(\theta_i)$ into $\langle\psi|$, $|\psi\rangle$.
Unitaries after $U(\theta_i)$ and observables are fused into $\widehat{Q}$.

Usually, the rotation gates used in QNN can be written in the form $U(\theta_i)=e^{-\frac{i}{2}\theta_i H}$. Here $H$ is the Hermitian generator of $U$ with only 2 unique eigenvalues +1 and -1.

In this way, the gradients of the circuit function $f$ with respect to $\theta_i$ are,
$$    \begin{aligned}
&\frac{\partial f(\theta)}{\partial \theta_i}=\frac{1}{2}\Big(f\big(\theta_+\big)-f\big(\theta_{-}\big)\Big), \\ &\theta_+=[\theta_1,\cdots,\theta_i+\frac{\pi}{2},\cdots,\theta_n], \theta_{-}=[\theta_1,\cdots,\theta_i-\frac{\pi}{2},\cdots,\theta_n],
\end{aligned}
$$
where $\theta_+$ and $\theta_{-}$ are the **positive shift** and **negative shift** of $\theta$.

Note that this parameter shift rule is **fundamentally different** from any numerical difference methods that only approximate the directional derivatives.
Instead, the equation calculates the **exact** gradient w.r.t $\theta_i$ without any approximation errors or numerical issues.

We apply $\text{softmax}$ on the expectation values of measurements $f(\theta)$ as the predicted probability for each class.
Then we calculate the cross entropy between the predicted probability distribution $p$ and the target distribution $t$ as the classification loss $\mathcal{L}$,
$$    \mathcal{L}(\theta)=-t^T\cdot\texttt{softmax}(f(\theta))=-\sum_{j=1}^m t_j \log{p_j},\quad p_j=\frac{e^{f_j(\theta)}}{\sum_{j=1}^m e^{f_j(\theta)}}.
$$

Then the gradient of the loss function with respect to $\theta_i$ is $\frac{\partial\mathcal{L}(\theta)}{\partial \theta_i}=\big(\frac{\partial\mathcal{L}(\theta)}{\partial f(\theta)}\big)^T\frac{\partial f(\theta)}{\partial \theta_i}$.

Here $\frac{\partial f(\theta)}{\partial \theta_i}$ can be calculated on real quantum circuit by the parameter shift rule, and $\frac{\partial\mathcal{L}(\theta)}{\partial f(\theta)}$ can be efficiently calculated on classical devices using backpropagation supported by automatic differentiation frameworks, e.g., PyTorch and TensorFlow.

Now we derive the parameter shift rule used in our QNN models.

Assume $U(\theta_i)=R_X(\theta_i),R_X(\alpha)=e^{-\frac{i}{2}\alpha X}$, where $X$ is the Pauli-X matrix.

Firstly, the RX gate is,
$$    \begin{aligned}
R_X(\alpha)&=e^{-\frac{i}{2}\alpha X}=\sum_{k=0}^{\infty}(-i\alpha/2)^kX^k/k!\\
&=\sum_{k=0}^{\infty}(-i\alpha/2)^{2k}X^{2k}/(2k)!+\sum_{k=0}^{\infty}(-i\alpha/2)^{2k+1}X^{2k+1}/(2k+1)!\\
&=\sum_{k=0}^{\infty}(-1)^k(\alpha/2)^{2k}I/(2k)!-i\sum_{k=0}^{\infty}(-1)^k(\alpha/2)^{2k+1}X/(2k+1)!\\
&=\cos(\alpha/2)I-i\sin(\alpha/2)X.
\end{aligned}
$$

Let $\alpha=\frac{\pi}{2}$, $R_X(\pm\frac{\pi}{2})=\frac{1}{\sqrt{2}}(I\mp iX)$.

As $f(\theta)=\langle\psi|R_X(\theta_i)^{\dagger}\widehat{Q}R_X(\theta_i)|\psi\rangle$, $R_X(\alpha)R_X(\beta)=R_X(\alpha+\beta)$, and $\frac{\partial}{\partial \alpha}R_X(\alpha)=-\frac{i}{2}XR_X(\alpha)$,
we have
$$\begin{aligned}
\frac{\partial f(\theta)}{\partial \theta_i}
% &=\langle\psi|\frac{\partial}{\partial \theta_i}R_X(\theta_i)^{\dagger}\widehat{Q}R_X(\theta_i)|\psi\rangle+\langle\psi|R_X(\theta_i)^{\dag}\widehat{Q}\frac{\partial}{\partial \theta_i}R_X(\theta_i)|\psi\rangle\\
=&\langle\psi|R_X(\theta_i)^{\dagger}(-\frac{i}{2}X)^{\dagger}\widehat{Q}R_X(\theta_i)|\psi\rangle+\langle\psi|R_X(\theta_i)^{\dagger}\widehat{Q}(-\frac{i}{2}X)R_X(\theta_i)|\psi\rangle\\
% &=\frac{1}{2}(\langle\psi|R_X(\theta_i)^{\dagger}(-iX)^{\dagger}\widehat{Q}R_X(\theta_i)|\psi\rangle+\langle\psi|R_X(\theta_i)^{\dagger}\widehat{Q}(-iX)R_X(\theta_i)|\psi\rangle)\\
=&\frac{1}{4}(\langle\psi|R_X(\theta_i)^{\dagger}(I-iX)^{\dagger}\widehat{Q}(I-iX)R_X(\theta_i)|\psi\rangle\\&-\langle\psi|R_X(\theta_i)^{\dagger}(I+iX)^{\dagger}\widehat{Q}(I+iX)R_X(\theta_i)|\psi\rangle)\\
=&\frac{1}{2}(\langle\psi|R_X(\theta_i)^{\dagger}R_X(\frac{\pi}{2})^{\dagger}\widehat{Q}R_X(\frac{\pi}{2})R_X(\theta_i)|\psi\rangle\\&-\langle\psi|R_X(\theta_i)^{\dagger}R_X(-\frac{\pi}{2})^{\dagger}\widehat{Q}R_X(-\frac{\pi}{2})R_X(\theta_i)|\psi\rangle)\\
=&\frac{1}{2}(f(\theta_+)-f(\theta_-)).
\end{aligned}
$$

Without loss of generality, the derivation holds for all unitaries of the form $e^{-\frac{i}{2}\alpha H}$, e.g., RX, RY, RZ, XX, YY, ZZ, where $H$ is a Hermitian matrix with only 2 unique eigenvalues +1 and -1.



