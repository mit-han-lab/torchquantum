import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

from torchpack.utils.logging import logger
from torchquantum.measurement import expval_obs_mat


__all__ = ["VQE"]


class VQE(object):
    def __init__(self, hamil, ansatz, train_configs) -> None:
        """Init function for VQE class
        Args:
            hamil (dict): A dictionary containing the information of the hamiltonian
         ansatz (Ansatz): An Ansatz object
        """
        self.hamil = hamil
        self.ansatz = ansatz
        self.train_configs = train_configs

        self.n_wires = hamil.n_wires
        self.n_epochs = self.train_configs.get("n_epochs", 100)
        self.n_steps = self.train_configs.get("n_steps", 10)
        self.optimizer_name = self.train_configs.get("optimizer", "Adam")
        self.scheduler_name = self.train_configs.get("scheduler", "CosineAnnealingLR")
        self.lr = self.train_configs.get("lr", 0.1)
        self.device = self.train_configs.get("device", "cpu")
        self.ansatz = self.ansatz.to(self.device)

    def get_expval(self, qdev):
        return expval_obs_mat(qdev, self.hamil.matrix.to(qdev.device))

    def get_loss(self):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=1,
            device=self.device,
        )
        self.ansatz(qdev)
        expval = self.get_expval(qdev)
        return expval
    
    def train(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(self.ansatz.parameters(), lr=self.lr)
        lr_scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_name)(optimizer, T_max=self.n_epochs)
        loss = None
        for epoch in range(self.n_epochs):
            for step in range(self.n_steps):
                loss = self.get_loss()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss}")
            lr_scheduler.step()
        return loss.detach().cpu().item()

# if __name__ == '__main__':
    # ansatz = Ansatz()
