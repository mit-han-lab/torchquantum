## Clifford QNN for MNIST-2 classification

Authors: Hanrui Wang, Gokul Subramanian Ravi

The model contains 16 encoder gates (4RY, 4RZ, 4RZ, 4RY) to encoder 16 pixels on 4 qubits. The encoder part is still non-clifford. The trainable part contains 
5 blocks. In each block, we have 4 RX, 4 RY, 4 RZ and then 4 CNOT with ring connections.

The advantange of using Clifford QNN is that the simulation of Clifford circuits are exponentially more efficient than simulation of general quantum circuits.

### TODOs
- [ ] Clifford encoder
- [ ] Improve quantization aware finetuning accuracy 

### Train the model in floating and then perform static quantization:

```python
python mnist_clifford_qnn.py
```

Train for 20 epochs. Test results:

|  Floating     | Accuracy  | Loss |
| ----------- | ----------- | --------- |
| Floating point      |  0.868      | 0.378 |
| Clifford | 0.660 | 0.648 |


### Train the model in floating and then perform quantization-aware finetuning:
Using the straight-through estimation (SSE) of gradients. 


```python
python mnist_clifford_qnn.py --finetune
```
Train for 20 epochs and then finetune 20 epochs. Test results:

|  Floating     | Accuracy  | Loss |
| ----------- | ----------- | --------- |
| Floating point    |  0.868      | 0.378 |
| Clifford | 0.722 | 0.582 |

