
# Quantum LSTM


Before the rise of the transformer, recurrent neural networks, especially Long Short-Term Memory (LSTM), were the most successful techniques for generating and analyzing sequential data. LSTM uses a combination of “memory” and “statefulness” tricks to understand which parts of the input are relevant to compute the output.


![LSTM architecture](https://github.com/MohammadrezaTavasoli/Quantum-LSTM/blob/master/figs/Structure-of-the-LSTM-cell-and-equations-that-describe-the-gates-of-an-LSTM-cell.png)

To convert classical LSTM to quantum-enhanced LSTM (QLSTM) [1] each of the classical linear operations Wf, Wi, WC, and Wo is replaced by a hybrid quantum-classical component that consists of a Variational Quantum Circuit sandwiched between classical layers.

## Authors

- [@MohammadrezaTavasoli](https://github.com/MohammadrezaTavasoli)


[1] [Chen, Samuel Yen-Chi, Shinjae Yoo, and Yao-Lung L. Fang. "Quantum long short-term memory." ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9747369)
