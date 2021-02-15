import pytorch_quantum as tq
import pytorch_quantum.functional as tqf

import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import pdb

logging.basicConfig(
    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.DEBUG)


class TrainableRxAll(tq.QuantumModule):
    """Rotation rx on all qubits
    The rotation angle is a parameter of each rotation gate
    One potential optimization is to compute the unitary of all gates
    together.
    """
    def __init__(self, n_gate: int):
        super().__init__()
        self.n_gate = n_gate
        self.gate_all = nn.ModuleList()
        for k in range(self.n_gate):
            self.gate_all.append(tq.RX(trainable=True))

    def forward(self, q_device: tq.QuantumDevice):
        # rx on all wires, assert the number of gate is the same as the number
        # of wires in the device.
        assert self.n_gate == q_device.n_wire, \
            f"Number of rx gates ({self.n_gate}) is different from number " \
            f"of wires ({q_device.n_wire})!"

        for k in range(self.n_gate):
            self.gate_all[k](q_device, wires=k)


class RxAll(tq.QuantumModule):
    """Rotation rx on all qubits
    The rotation angle is from input activation
    """
    def __init__(self, n_gate: int):
        super().__init__()
        self.n_gate = n_gate
        self.gate_all = nn.ModuleList()
        for k in range(self.n_gate):
            self.gate_all.append(tq.RX())

    def forward(self, x, q_device: tq.QuantumDevice):
        # rx on all wires, assert the number of gate is the same as the number
        # of wires in the device.
        assert self.n_gate == q_device.n_wire, \
            f"Number of rx gates ({self.n_gate}) is different from number " \
            f"of wires ({q_device.n_wire})!"

        for k in range(self.n_gate):
            self.gate_all[k](q_device, wires=k, params=x[:, k])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.sigmoid = nn.Sigmoid()
        self.q_device0 = tq.QuantumDevice(n_wire=10)
        self.q_layer0 = TrainableRxAll(n_gate=10)
        self.q_layer1 = RxAll(n_gate=10)
        self.q_layer2 = tq.RX(has_params=True,
                              trainable=False,
                              init_params=-np.pi / 4)

        self.q_layer3 = tq.RZ(has_params=True,
                              trainable=True)
        self.q_device1 = tq.QuantumDevice(n_wire=3)
        self.q_layer4 = tq.CY()
        self.q_layer5 = tq.Toffoli()
        self.q_layer6 = tq.PhaseShift(has_params=True,
                                      trainable=True)
        self.q_layer7 = tq.Rot(has_params=True,
                               trainable=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * 2 * np.pi

        self.q_device0.reset_states(x.shape[0])
        self.q_device1.reset_states(x.shape[0])
        self.q_layer0(self.q_device0)
        self.q_layer1(x, self.q_device0)
        tqf.rx(self.q_device0, 1, x[:, 1])
        self.q_layer2(self.q_device0, wires=5)
        tqf.ry(self.q_device0, 2, x[:, 2])
        tqf.rz(self.q_device0, 3, x[:, 3])
        tqf.s(self.q_device0, 4)
        tqf.t(self.q_device0, 5)
        self.q_layer3(self.q_device0, wires=6)
        tqf.sx(self.q_device0, 7)
        tqf.x(self.q_device1, wires=0)
        tqf.cnot(self.q_device1, wires=[0, 2])
        tqf.cnot(self.q_device1, wires=[0, 1])
        tqf.cnot(self.q_device1, wires=[2, 0])
        tqf.cz(self.q_device0, wires=[0, 5])
        tqf.cnot(self.q_device0, wires=[0, 5])
        tqf.cy(self.q_device0, wires=[0, 5])
        self.q_layer4(self.q_device0, wires=[3, 8])
        tqf.swap(self.q_device0, wires=[2, 3])
        tqf.cswap(self.q_device0, wires=[4, 5, 6])
        self.q_layer5(self.q_device0, wires=[8, 5, 0])
        self.q_layer6(self.q_device0, wires=8)
        tqf.phaseshift(self.q_device0, 7, x[:, 7])
        self.q_layer7(self.q_device0, wires=4)
        tqf.rot(self.q_device0, 5, x[:, 6:9])

        x = tq.expval(self.q_device0, list(range(10)), [tq.PauliY()] * 10)

        output = F.log_softmax(x, dim=1)

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        # parameter update
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging '
                             'training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--pdb', action='store_true', default=False,
                        help='pdb')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.pdb:
        pdb.set_trace()
    # use_cuda = False

    torch.manual_seed(args.seed)
    logging.info(f"setting device!")

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
