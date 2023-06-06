import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchquantum as tq
import torchquantum.functional as tqf
import argparse
import tqdm
import time

# Preparing the entangled state/ 2 qubit bell pair.
def bell_pair():
  qdev = tq.QuantumDevice(n_wires=2, bsz=1, device="cpu")
  qdev.h(wires=0)
  qdev.cnot(wires=[0, 1])
  return qdev

# Encoding the message
def encode_message(qdev, qubit, msg):
    if len(msg) != 2 or not set(msg).issubset({"0","1"}):
        raise ValueError(f"message '{msg}' is invalid")
    if msg[1] == "1":
        qdev.x(wires=qubit)
    if msg[0] == "1":
        qdev.z(wires=qubit)
    return qdev

# Decoding the message
def decode_message(qdev):
    qdev.cx(wires=[0, 1])
    qdev.h(wires=0)
    return qdev

# Putting all these functions together
def main():
    # Creating the entangled pair between Alice and Bob
    qdev = bell_pair()
    # Encoding the message at Alice's end
    message = '10'
    qdev = encode_message(qdev, 1, message)
    # Decoding the original message at Bob's end
    qdev = decode_message(qdev) 
    # Finally, Bob measures his qubits to read Alice's message
    print(tq.measure(qdev, n_shots=1024))

if __name__ == "__main__":
    main()