Here we provide 6 examples for 2 different machines, 2 tasks and 2 design spaces.

The table shows the expected results for these examples.

| examples id |   machine  |    task    | design space | expected accuracy |
|:-----------:|:----------:|:----------:|:------------:|:-----------------:|
|      0      | IBMQ_Quito | MNIST-0123 |    U3+CU3    |        71%        |
|      1      |  IBMQ_Lima | MNIST-0123 |    U3+CU3    |       55.3%       |
|      2      | IBMQ_Quito | FASHION-36 |    U3+CU3    |        88%        |
|      3      |  IBMQ_Lima | FASHION-36 |    U3+CU3    |       88.7%       |
|      4      | IBMQ_Quito | FASHION-36 |    RZZ+RY    |       87.7%       |
|      5      |  IBMQ_Lima | FASHION-36 |    RZZ+RY    |       88.7%       |

For example, if you want to run example2, you only need to run the following command lines

```bash
bash artifact/example2/QuantumNas/1_train_supercircuit.sh
bash artifact/example2/QuantumNas/2_search.sh
bash artifact/example2/QuantumNas/3_train_subcircuit.sh
bash artifact/example2/QuantumNas/4_prune.sh
bash artifact/example2/QuantumNas/5_eval.sh
```