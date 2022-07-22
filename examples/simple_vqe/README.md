#### Simple VQE example
## Usage
Write the hamiltonian information in a .txt file such as `h2.txt`
The first line is `molecule name`, `transformation`, `number of qubits`

```python
python simple_vqe.py --hamil_filename=h2.txt
```

Optionally add more configs:
```python
python simple_vqe.py --hamil_filename=h2.txt --epochs=100 --steps_per_epoch=100 --n_blocks=3
```

