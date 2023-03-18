# predict_quantum_acc
## Example
Please run
```bash
$ python train.py huge/default
```
as a simple example.
## Data_set
The data_set is stored in raw_data_qasm folder, each data is a list of 3 elements.
```python
data_set = pickle.load(file)
data= data_set[0]
# data[0]: String, the qasm for the circuit,
# data[1]: Dict, contains the noise information
# data[2]: Float, probability of successful trials.
```
## Albation Studies
We study the effect of each feature on the performance of the model.
```bash
python train.py huge/default # Default training
# Ablation studies
python train.py huge/layer1 # 1 layer of transformer
python train.py huge/layer3 # 3 layers of transformer
python train.py huge/onlygf # Only use global features
python train.py huge/wogateerror # No gate error
python train.py huge/wogateindex # No gate index
python train.py huge/wogatetype # No gate type
python train.py huge/wogf # No global features
python train.py huge/woqubitindex # No qubit index
python train.py huge/wot1t2 # No t1 and t2
```
## Environment
The environment is as follows:
```bash
torch == 1.13.0
Torch-geometric == 2.2.0
Qiskit == 0.39.4
Python == 3.10.8
```
