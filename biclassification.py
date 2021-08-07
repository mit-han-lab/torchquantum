import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np

import torchquantum as tq
import torchquantum.functional as tqf

from examples.core.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR
from qiskit import IBMQ
from torchquantum.plugins import QiskitProcessor
from torch.utils.tensorboard import SummaryWriter   

class Classification2Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = []
        self.target = []
        sum0 = 0
        sum1 = 0
        for x in np.linspace(0, 1, num=11):
            for y in np.linspace(0, 1, num=11):
                self.data.append(torch.tensor([x, y]))
                if (x**2 + y**2 <= 0.55**2 or (x-1)**2 + (y-1)**2 <= 0.55**2):
                    self.target.append(1)
                    sum1 = sum1 + 1
                else:
                    self.target.append(0)
                    sum0 = sum0 + 1
            print(self.target[-11:])

    def __getitem__(self, idx):
        return {'data': self.data[idx], 'target': self.target[idx]}

    def __len__(self):
        return len(self.target)


class SimpleModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 5

            # gates with trainable parameters
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.ry1 = tq.RY(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
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
            self.ry0(self.q_device, wires=2)
            tqf.cnot(self.q_device, wires=[0, 1], static=self.static_mode, parent_graph=self.graph)
            tqf.x(self.q_device, wires=0, static=self.static_mode, parent_graph=self.graph)
            tqf.ccnot(self.q_device, wires=[0, 2, 1], static=self.static_mode, parent_graph=self.graph)
            self.ry1(self.q_device, wires=4)
            tqf.ccnot(self.q_device, wires=[1, 4, 3], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 5
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder([{'input_idx': [0], 'func': 'ry', 'wires': [0]}])

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = 2 * torch.arcsin(torch.sqrt(x.sum(dim=1) - 2 * x[:,0] * x[:,1]))
        x = x.view(bsz, -1)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x, False)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = (x.reshape(bsz, self.n_wires)[:, 3] + 1) / 2

        return x


class SimpleModel2(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 5

            # gates with trainable parameters
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.ry1 = tq.RY(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
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
            self.ry0(self.q_device, wires=2)
            tqf.cnot(self.q_device, wires=[0, 1], static=self.static_mode, parent_graph=self.graph)
            tqf.x(self.q_device, wires=0, static=self.static_mode, parent_graph=self.graph)
            tqf.ccnot(self.q_device, wires=[0, 2, 1], static=self.static_mode, parent_graph=self.graph)
            self.ry1(self.q_device, wires=4)
            tqf.ccnot(self.q_device, wires=[1, 4, 3], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 5
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder([{'input_idx': [0], 'func': 'ry', 'wires': [0]}])

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = 2 * torch.arcsin(torch.sqrt(x.sum(dim=1) - 2 * x[:,0] * x[:,1]))
        x = x.view(bsz, -1)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x, False)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, self.n_wires)[:, [1, 3]]

        return x


class SimpleModel3(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 2

            # gates with trainable parameters
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.ry1 = tq.RY(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
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
            self.ry0(self.q_device, wires=0)
            self.ry1(self.q_device, wires=1)
            tqf.cnot(self.q_device, wires=[0, 1], static=self.static_mode, parent_graph=self.graph)
            tqf.x(self.q_device, wires=0, static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder([{'input_idx': [0], 'func': 'ry', 'wires': [0]}])

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.use_params_shift = False
    
    def use_params_shift(self, use_params_shift):
        self.use_params_shift = use_params_shift

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = 2 * torch.arcsin(torch.sqrt(x.sum(dim=1) - 2 * x[:,0] * x[:,1]))
        x = x.view(bsz, -1)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x, False)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, self.n_wires)[:, :]
        return x
    
    def shift_and_run(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = 2 * torch.arcsin(torch.sqrt(x.sum(dim=1) - 2 * x[:,0] * x[:,1]))
        x = x.view(bsz, -1)

        if use_qiskit:
            with torch.no_grad():
                x = self.qiskit_processor.process_parameterized_and_shift(
                    self.q_device, self.encoder, self.q_layer, self.measure, x, False)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(1 + 2 * len(list(self.q_layer.named_parameters())), bsz, self.n_wires)

        return x

def train(dataflow, model, device, optimizer, criterion, use_params_shift):
    for feed_dict in dataflow['train']:
        inputs = feed_dict['data'].to(device)
        targets = feed_dict['target'].to(device)

        if use_params_shift:
            grad_list, loss = get_gradient(model, optimizer, criterion, inputs, targets)
            optimizer.zero_grad()
            for named_param, grad in zip(model.named_parameters(), grad_list):
                named_param[-1].grad = torch.reshape(grad.to(dtype=torch.float32), [1, 1])
            optimizer.step()
        else:
            outputs = model(inputs, use_qiskit=False)
            loss = criterion(F.log_softmax(outputs, dim=1), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"loss: {loss.item()}", end='\r')


def get_gradient(model, optimizer, criterion, inputs, targets):
    results = model.shift_and_run(inputs, use_qiskit=True)
    outputs = results[0,:,:]
    outputs.requires_grad=True
    loss = criterion(F.log_softmax(outputs, dim=1), targets)

    optimizer.zero_grad()
    loss.backward()
    outputs_grad = outputs.grad

    with torch.no_grad():
        grad_params_out_list = []
        cnt = 0
        for named_param in model.named_parameters():
            cnt = cnt + 1
            out1 = results[cnt,:,:]
            cnt = cnt + 1
            out2 = results[cnt,:,:]
            grad_params_out = 0.5 * (out1 - out2)
            grad_params_out_list.append(grad_params_out)

        grad_params_loss_list2 = []
        for grad_params_out in grad_params_out_list:
            grad_params_out = torch.sum(grad_params_out * outputs_grad)
            grad_params_out = torch.sum(grad_params_out)
            grad_params_loss_list2.append(grad_params_out)
        return grad_params_loss_list2, loss



def valid_test(dataflow, split, model, device, criterion, epoch, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['data'].to(device)
            targets = feed_dict['target'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    output_all = F.log_softmax(output_all, dim=1)
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()


    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")
    return loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true', help='compute with '
                                                              'static mode')
    parser.add_argument('--wires-per-block', type=int, default=2,
                        help='wires per block int static mode')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of training epochs')

    args = parser.parse_args()

    mydataset = Classification2Dataset()
    dataset = {'train': mydataset, 'valid': mydataset, 'test': mydataset}
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=15,
            sampler=sampler,
            num_workers=8,
            pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = SimpleModel3().to(device)
    processor_simulation = QiskitProcessor(use_real_qc=True, backend_name='ibmq_santiago')
    model.set_qiskit_processor(processor_simulation)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=3e-1, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = torch.nn.NLLLoss()

    if args.static:
        # optionally to switch to the static mode, which can bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    writer = SummaryWriter('./realqc_training_realqc_eval_result/')
    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        train(dataflow, model, device, optimizer, criterion, use_params_shift=True)
        print(optimizer.param_groups[0]['lr'])

        # valid
        loss, accuracy = valid_test(dataflow, 'valid', model, device, criterion, epoch=epoch, qiskit=True)
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('accuracy', accuracy, epoch)

        scheduler.step()

    # run on Qiskit simulator and real Quantum Computers
    # try:

        # firstly perform simulate
        # print(f"\nTest with Qiskit Simulator")
        # processor_simulation = QiskitProcessor(use_real_qc=True, backend_name='ibmq_santiago')
        # model.set_qiskit_processor(processor_simulation)
        # valid_test(dataflow, 'test', model, device, criterion, qiskit=True)

        # then try to run on REAL QC
        # backend_name = 'ibmqx2'
        # print(f"\nTest on Real Quantum Computer {backend_name}")
        # processor_real_qc = QiskitProcessor(use_real_qc=True,
        #                                     backend_name=backend_name)
        # model.set_qiskit_processor(processor_real_qc)
        # valid_test(dataflow, 'test', model, device, qiskit=True)
    # except ImportError:
    #     print("Please install qiskit, create an IBM Q Experience Account and "
    #           "save the account token according to the instruction at "
    #           "'https://github.com/Qiskit/qiskit-ibmq-provider', "
    #           "then try again.")


if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    torch.manual_seed(10)
    main()
