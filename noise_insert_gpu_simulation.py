# trainer:
#   name: q_noise_aware_trainer
#   add_noise: True
#   act_quant: False
#   act_quant_skip_last_node: True
#   act_quant_loss: True
#   act_quant_loss_lambda: 10
#   act_quant_upper_bound: 2
#   act_quant_lower_bound: -2


# qiskit:
#   use_qiskit: False
#   use_real_qc: False
#   backend_name: None
#   noise_model_name: ibmq_santiago
#   n_shots: 8192
#   initial_layout: [0, 1, 2, 3]
#   optimization_level: 2
#   max_jobs: 1
#   transpile_with_ancilla: False

# model:
#   transpile_before_run: True
#   load_op_list: False

import torch
import argparse
import torchquantum as tq
import torch.optim as optim
import torch.nn.functional as F
import torchquantum.functional as tqf
from examples.core.datasets import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR
from examples.core.models.super_models import model_dict
from examples.core.schedulers import CosineAnnealingWarmupRestarts
from torchquantum.plugins import tq2qiskit, qiskit2tq

from dst import *
from torchquantum.noise_model import *
from torchquantum.utils import (build_module_from_op_list,
                                build_module_op_list,
                                get_v_c_reg_mapping,
                                get_p_c_reg_mapping,
                                get_p_v_reg_mapping,
                                get_cared_configs)
from torchquantum.plugins import QiskitProcessor


def train(dataflow, model, device, optimizer, masker):
    for feed_dict in dataflow['train']:
        inputs = feed_dict['image'].to(device)
        targets = feed_dict['digit'].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # masker.step()
        print(f"loss: {loss.item()}", end='\r')

def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

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

class QFCModel(tq.QuantumModule):

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4

            # gates with trainable parameters
            self.layer1 = tq.QuantumModuleList()
            self.layer2 = tq.QuantumModuleList()
            self.layer3 = tq.QuantumModuleList()
            self.layer4 = tq.QuantumModuleList()
            self.clayer1 = tq.QuantumModuleList()
            self.clayer2 = tq.QuantumModuleList()
            self.clayer3 = tq.QuantumModuleList()
            self.clayer4 = tq.QuantumModuleList()

            for idx in range(8):
                self.layer1.append(tq.U3(has_params=True, trainable=True))
                self.layer2.append(tq.U3(has_params=True, trainable=True))
                self.layer3.append(tq.U3(has_params=True, trainable=True))
                self.layer4.append(tq.U3(has_params=True, trainable=True))
                self.clayer1.append(tq.CU3(has_params=True, trainable=True))
                self.clayer2.append(tq.CU3(has_params=True, trainable=True))
                self.clayer3.append(tq.CU3(has_params=True, trainable=True))
                self.clayer4.append(tq.CU3(has_params=True, trainable=True))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):

            self.q_device = q_device

            for idx in range(8):
                self.layer1[idx](self.q_device, wires=0)
                self.layer2[idx](self.q_device, wires=1)
                self.layer3[idx](self.q_device, wires=2)
                self.layer4[idx](self.q_device, wires=3)
                self.clayer1[idx](self.q_device, wires=[0,1])
                self.clayer2[idx](self.q_device, wires=[1,2])
                self.clayer3[idx](self.q_device, wires=[2,3])
                self.clayer4[idx](self.q_device, wires=[3,0])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.num_class = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        if self.num_class == 2:
            x = x.reshape(bsz, 2, 2).sum(-1).squeeze()

        x = F.log_softmax(x, dim=1)

        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true', help='compute with '
                                                              'static mode')
    parser.add_argument('--wires-per-block', type=int, default=2,
                        help='wires per block int static mode')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--death_rate', type=float, default=0.1)
    parser.add_argument('--death_mode', type=str, default='magnitude')
    parser.add_argument('--growth_mode', type=str, default='random')
    parser.add_argument('--density', type=float, default=1)
    parser.add_argument('--num_class', type=int, default=4)

    args = parser.parse_args()

    if args.num_class == 2:
        dataset = MNIST(
            root='./mnist_data',
            train_valid_split_ratio=[0.95, 0.05],
            center_crop=24,
            resize=28,
            resize_mode='bilinear',
            binarize=False,
            digits_of_interest=[3, 6],
        )
    elif args.num_class == 4:
        dataset = MNIST(
            root='./mnist_data',
            train_valid_split_ratio=[0.95, 0.05],
            center_crop=24,
            resize=28,
            resize_mode='bilinear',
            binarize=False,
            digits_of_interest=[0,1,2,3],
        )
    else:
        assert False

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

    model = QFCModel()
    model.num_class = args.num_class

    cnt = 0
    train_cnt = 0
    for p in model.q_layer.parameters():
        cnt += p.nelement()
        if p.requires_grad == True:
            train_cnt += p.nelement()
    print(cnt, train_cnt)


    processor = QiskitProcessor(
        use_real_qc=False,
        backend_name=None,
        noise_model_name='ibmq_santiago',
        coupling_map_name='ibmq_santiago',
        basis_gates_name='ibmq_santiago',
        n_shots=8192,
        initial_layout=[0,1,2,3],
        seed_transpiler=42,
        seed_simulator=42,
        optimization_level=2,
        max_jobs=1,
        remove_ops=False,
        remove_ops_thres=None,
        transpile_with_ancilla=False,
        hub=None,
        layout_method=None,
        routing_method=None,
    )

    circ = tq2qiskit(model.q_device, model.q_layer)
    circ.measure(list(range(model.q_device.n_wires)),
                list(range(model.q_device.n_wires)))
    circ_transpiled = processor.transpile(circs=circ)
    q_layer = qiskit2tq(circ=circ_transpiled)
    model.measure.set_v_c_reg_mapping(
        get_v_c_reg_mapping(circ_transpiled))
    model.q_layer = q_layer
    
    cnt = 0
    train_cnt = 0
    for p in model.q_layer.parameters():
        cnt += p.nelement()
        if p.requires_grad == True:
            train_cnt += p.nelement()
    print(cnt, train_cnt)

    noise_model_tq = noise_model_tq = NoiseModelTQ(
        noise_model_name='ibmq_santiago',
        n_epochs=args.epochs,
        noise_total_prob=None,
        ignored_ops=['id', 'kraus', 'reset'],
        prob_schedule=None,
        prob_schedule_separator=None,
        factor=1,
        add_thermal=True
    )
    noise_model_tq.is_add_noise = True
    noise_model_tq.v_c_reg_mapping = get_v_c_reg_mapping(
        circ_transpiled)
    noise_model_tq.p_c_reg_mapping = get_p_c_reg_mapping(
        circ_transpiled)
    noise_model_tq.p_v_reg_mapping = get_p_v_reg_mapping(
        circ_transpiled)
    model.set_noise_model_tq(noise_model_tq)

    model.to(device)


    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    # scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=n_epochs,
        max_lr=5e-3,
        min_lr=0,
        warmup_steps=30,
    )

    # decay = CosineDecay(args.death_rate, len(dataflow['train']) * (args.epochs))
    # masker = Masking(
    #     optimizer = optimizer,
    #     death_rate=args.death_rate,
    #     death_rate_decay=decay, 
    #     death_mode=args.death_mode, 
    #     growth_mode=args.growth_mode,
    #     args=args,
    #     decay_schedule='cosine'
    # )
    # masker.add_module(model, args.density, sparse_init='uniform_global')
    masker = None


    if args.static:
        # optionally to switch to the static mode, which can bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")

        # # set noise model tq to train
        # model.noise_model_tq.mode = 'eval'
        # model.noise_model_tq.adjust_noise(epoch)

        train(dataflow, model, device, optimizer, masker)
        print(optimizer.param_groups[0]['lr'])

        # if args.density < 1:
        #     if epoch % 8 == 0:
        #         masker.at_end_of_epoch()

        # reset 
        # model.noise_model_tq.mode = 'eval'

        # valid
        valid_test(dataflow, 'valid', model, device)
        valid_test(dataflow, 'test', model, device, qiskit=False)
        scheduler.step()

    # test
    
    # # run on Qiskit simulator and real Quantum Computers
    # try:
    #     from qiskit import IBMQ
    #     from torchquantum.plugins import QiskitProcessor

    #     # firstly perform simulate
    #     print(f"\nTest with Qiskit Simulator")
    #     processor_simulation = QiskitProcessor(use_real_qc=False)
    #     model.set_qiskit_processor(processor_simulation)
    #     valid_test(dataflow, 'test', model, device, qiskit=True)

    #     # then try to run on REAL QC
    #     backend_name = 'ibmq_quito'
    #     print(f"\nTest on Real Quantum Computer {backend_name}")
    #     processor_real_qc = QiskitProcessor(use_real_qc=True,
    #                                         backend_name=backend_name)
    #     model.set_qiskit_processor(processor_real_qc)
    #     valid_test(dataflow, 'test', model, device, qiskit=True)
    # except ImportError:
    #     print("Please install qiskit, create an IBM Q Experience Account and "
    #           "save the account token according to the instruction at "
    #           "'https://github.com/Qiskit/qiskit-ibmq-provider', "
    #           "then try again.")



if __name__ == '__main__':
    main()












