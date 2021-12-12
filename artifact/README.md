Here we provide 6 examples for 2 different machines, 2 tasks and 2 design spaces.

The table shows the expected results for these examples.

| examples id |   machine  |    task    | design space | expected accuracy |
|:-----------:|:----------:|:----------:|:------------:|:-----------------:|
|      2      | IBMQ_Quito | MNIST-0123 |    U3+CU3    |        74%        |
|      3      |  IBMQ_Lima | MNIST-0123 |    U3+CU3    |       55.3%       |
|      4      | IBMQ_Quito | FASHION-36 |    U3+CU3    |        90%        |
|      5      |  IBMQ_Lima | FASHION-36 |    U3+CU3    |       88.7%       |
|      6      | IBMQ_Quito | FASHION-36 |    ZZ+RY     |        89%        |
|      7      |  IBMQ_Lima | FASHION-36 |    ZZ+RY     |       88.7%       |

For example, if you want to run example2, you only need to follow the instructions below.

However, if you want to run any other experiments, you can refer to our colab [link](https://colab.research.google.com/drive/1WJDzqC6Zt11uQF_j9yiAs36uDs-Uzm1d?usp=sharing) for general experiments.

## Installation
```bash
git clone https://github.com/Hanrui-Wang/pytorch-quantum.git
cd pytorch-quantum
pip install --editable .
pip install pathos
pip install tensorflow_model_optimization
export PYTHONPATH=.
```

## Fix qiskit's bug
Now your qiskit version should be 0.32.1. Modify the part after line 346 of `lib/python3.x/site-packages/qiskit/providers/aer/backends/aerbackend.py` from this:
```python
elif parameter_binds:
    # Handle parameter binding
    parameterizations = self._convert_binds(circuits, parameter_binds)
    assemble_binds = []
    assemble_binds.append({param: 1 for bind in parameter_binds for param in bind})
    qobj = assemble(circuits, self, parameter_binds=assemble_binds,
                    parameterizations=parameterizations)
```
to this:
```python
elif parameter_binds:
            # Handle parameter binding
            # parameterizations = self._convert_binds(circuits, parameter_binds)
            # assemble_binds = []
            # assemble_binds.append({param: 1 for bind in parameter_binds for param in bind})

            qobj = assemble(circuits, self, parameter_binds=parameter_binds)
```

## Store the token
Next entor the following code into the python interpreter to store a qiskit token to your local file. You can replace it with your own token from your IBMQ account.
```python
from qiskit import IBMQ
IBMQ.save_account('0238b0afc0dc515fe7987b02706791d1719cb89b68befedc125eded0607e6e9e9f26d3eed482f66fdc45fdfceca3aab2edb9519d96b39e9c78040194b86e7858', overwrite=True)
```

## Run QuantumNAS
```bash
bash artifact/example2/QuantumNas/1_train_supercircuit.sh
bash artifact/example2/QuantumNas/2_search.sh
bash artifact/example2/QuantumNas/3_train_subcircuit.sh
bash artifact/example2/QuantumNas/4_prune.sh
bash artifact/example2/QuantumNas/5_eval.sh
```