# Quantum Circuit Born Machine

Quantum Circuit Born Machine (QCBM) [1] is a generative modeling algorithm which uses Born rule from quantum mechanics to sample from a quantum state $|\psi \rangle$ learned by training an ansatz $U(\theta)$ [1][2]. In this tutorial we show how `torchquantum` can be used to model a Gaussian mixture with QCBM.

## References

1. Liu, Jin-Guo, and Lei Wang. “Differentiable learning of quantum circuit born machines.” Physical Review A 98.6 (2018): 062324.
2. Gili, Kaitlin, et al. "Do quantum circuit born machines generalize?." Quantum Science and Technology 8.3 (2023): 035021.