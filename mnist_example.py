import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

import torchquantum as tq
import torchquantum.functional as tqf
from examples.core.datasets import MNIST


class QFCModel(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.encoder = tq.MultiPhaseEncoder([tqf.rx] * 4 + [tqf.ry] * 4 +
                                                [tqf.rz] * 4 + [tqf.rx] * 4)
            self.random_layer = tq.RandomLayer(n_ops=50,
                                               wires=list(range(self.n_wires)))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x):
            """
            1. To convert tq QuantumModule to qiskit or run in the static
            model, need to:
                (1) add @tq.static_support before the forward
                (2) make sure to add
                    static=self.static_mode and
                    parent_graph=self.graph
                    to all the tqf functions, such as tqf.hadamard below
            """
            self.q_device = q_device

            self.encoder(self.q_device, x)
            self.random_layer(self.q_device)
            tqf.hadamard(self.q_device, wires=1, static=self.static_mode,
                         parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        self.q_layer(self.q_device, x)

        x = self.measure(self.q_device).reshape(bsz, 2, 2)
        x = x.sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x

    def forward_qiskit(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        measured = self.q_layer.qiskit_processor.process(
            self.q_device, self.q_layer, x)
        measured = measured.reshape(bsz, 2, 2)

        x = measured.sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)

        return x


def train(dataflow, model, device, optimizer):
    for feed_dict in dataflow['train']:
        inputs = feed_dict['image'].to(device)
        targets = feed_dict['digit'].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}", end='\r')


def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)
            if qiskit:
                outputs = model.forward_qiskit(inputs)
            else:
                outputs = model(inputs)
            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")


def main():
    # the torchvision has some bugs on downloading MNIST, so download here
    # manually
    if not os.path.exists('./mnist_data'):
        os.system('wget hanlab.mit.edu/files/quantum/torchquantum'
                  '/mnist_data.tar.gz')
        os.system('tar -xzvf mnist_data.tar.gz')

    dataset = MNIST(
        root='./mnist_data',
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[3, 6],
        n_test_samples=75,
    )
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=256,
            sampler=sampler,
            num_workers=8,
            pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = QFCModel().to(device)

    n_epochs = 30
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    static_mode = True
    if static_mode:
        # optionally to switch to the static mode, which may bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=2)

    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        train(dataflow, model, device, optimizer)
        print(optimizer.param_groups[0]['lr'])

        # valid
        valid_test(dataflow, 'valid', model, device)
        scheduler.step()

    # test
    valid_test(dataflow, 'test', model, device, qiskit=False)

    # run on Qiskit simulator and real Quantum Computers
    try:
        from qiskit import IBMQ
        from torchquantum.plugins import QiskitProcessor

        # firstly perform simulate
        print(f"\nTest with Qiskit Simulator")
        processor_simulation = QiskitProcessor(use_real_qc=False)
        model.q_layer.set_qiskit_processor(processor_simulation)
        valid_test(dataflow, 'test', model, device, qiskit=True)

        # then try to run on REAL QC
        backend_name = 'ibmq_santiago'
        print(f"\nTest on Real Quantum Computer {backend_name}")
        processor_real_qc = QiskitProcessor(use_real_qc=True,
                                            backend_name=backend_name)
        model.q_layer.set_qiskit_processor(processor_real_qc)
        valid_test(dataflow, 'test', model, device, qiskit=True)
    except ImportError:
        print("Please install qiskit and create an IBM Q Experience Account "
              "(it's free!), then try again")


if __name__ == '__main__':
    main()
