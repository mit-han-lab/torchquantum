import matplotlib.pyplot as plt
import numpy as np
import torch
from torchquantum.algorithm import QCBM, MMDLoss
import torchquantum as tq


# Function to create a gaussian mixture
def gaussian_mixture_pdf(x, mus, sigmas):
	mus, sigmas = np.array(mus), np.array(sigmas)
	vars = sigmas**2
	values = [
		(1 / np.sqrt(2 * np.pi * v)) * np.exp(-((x - m) ** 2) / (2 * v))
		for m, v in zip(mus, vars)
	]
	values = np.sum([val / sum(val) for val in values], axis=0)
	return values / np.sum(values)

# Create a gaussian mixture
n_wires = 6
x_max = 2**n_wires
x_input = np.arange(x_max)
mus = [(2 / 8) * x_max, (5 / 8) * x_max]
sigmas = [x_max / 10] * 2
data = gaussian_mixture_pdf(x_input, mus, sigmas)

# This is the target distribution that the QCBM will learn
target_probs = torch.tensor(data, dtype=torch.float32)

# Ansatz
layers = tq.RXYZCXLayer0({"n_blocks": 6, "n_wires": n_wires, "n_layers_per_block": 1})

qcbm = QCBM(n_wires, layers)

# To train QCBMs, we use MMDLoss with radial basis function kernel.
bandwidth = torch.tensor([0.25, 60])
space = torch.arange(2**n_wires)
mmd = MMDLoss(bandwidth, space)

# Optimization
optimizer = torch.optim.Adam(qcbm.parameters(), lr=0.01)
for i in range(100):
	optimizer.zero_grad(set_to_none=True)
	pred_probs = qcbm()
	loss = mmd(pred_probs, target_probs)
	loss.backward()
	optimizer.step()
	print(i, loss.item())

# Visualize the results
with torch.no_grad():
	pred_probs = qcbm()

plt.plot(x_input, target_probs, linestyle="-.", label=r"$\pi(x)$")
plt.bar(x_input, pred_probs, color="green", alpha=0.5, label="samples")
plt.xlabel("Samples")
plt.ylabel("Prob. Distribution")

plt.legend()
plt.show()
