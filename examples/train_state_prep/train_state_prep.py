# import os
# import pickle
# from qiskit import QuantumCircuit
# from qiskit.providers.fake_provider import *
# from rand_circ_native import *
# from qiskit import IBMQ
# import sys
#
# dir_path = './application_qasm'
# files = os.listdir(dir_path)
#
# # ini_backend = FakeGuadalupe()
#
# def IBMQ_ini(backend_str):
#     IBMQ.load_account()
#     # provider = IBMQ.get_provider(hub="ibm-q-research", group="MIT-1", project="main")
#     provider = IBMQ.get_provider(hub='ibm-q-ornl', group='anl', project='csc428')
#     backend = provider.get_backend(backend_str)
#     return backend
#
# def simu(circ,backend,n_used, shots):
#     result = backend.run(circ,shots=shots).result()
#     counts = result.get_counts()
#     fidelity = counts[n_used * "0"] / shots
#     return fidelity
#
#
# for file_name in files:
#     data = []
#
#     circ_app = QuantumCircuit.from_qasm_file(dir_path+'/'+file_name)
#     if(circ_app.num_qubits>7):
#         # ini_backend = FakeGuadalupe()
#         continue
#     else:
#         # ini_backend = FakeJakarta()
#         backend = IBMQ_ini(sys.argv[1])
#
#     for i in range(50):
#         props = backend.properties().to_dict()
#         my_dict = build_my_noise_dict(props)
#         # my_dict = get_randomized_mydict(my_dict)
#         # backend = get_modified_backend(ini_backend, my_dict)
#         # simulator = AerSimulator.from_backend(backend)
#
#         circ_app.remove_final_measurements()
#         transpiled = transpile(circ_app, backend)
#         appended = circ_app.compose(circ_app.inverse())
#         appended.measure_active()
#         appended = transpile(appended,backend)
#
#         try:
#             fidelity = simu(
#                 appended, backend, appended.num_clbits, shots=4096
#             )
#         except:
#             print("bypassed")
#
#         data_p = [transpiled,my_dict,fidelity]
#         data.append(data_p)
#         # print(fidelity)
#     file_open = open('./app_real_data_test/'+sys.argv[1]+file_name[:-4]+'data','wb')
#     pickle.dump(data,file_open)
#     file_open.close()

import torch
import torch.optim as optim
import argparse

import torchquantum as tq
from torch.optim.lr_scheduler import CosineAnnealingLR

import random
import numpy as np


class QModel(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.cu3_0 = tq.CU3(has_params=True, trainable=True)
        self.cu3_1 = tq.CU3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)

    def forward(self, q_device: tq.QuantumDevice):
        q_device.reset_states(1)
        self.u3_0(q_device, wires=0)
        self.u3_1(q_device, wires=1)
        self.cu3_0(q_device, wires=[0, 1])
        self.u3_2(q_device, wires=0)
        self.u3_3(q_device, wires=1)
        self.cu3_1(q_device, wires=[1, 0])


def train(target_state, device, model, optimizer):
    model(device)
    result_state = device.get_states_1d()[0]

    # compute the state infidelity
    loss = 1 - torch.dot(result_state, target_state).abs() ** 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(
        f"infidelity (loss): {loss.item()}, \n target state : "
        f"{target_state.detach().cpu().numpy()}, \n "
        f"result state : {result_state.detach().cpu().numpy()}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=20000, help="number of training epochs"
    )

    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = QModel().to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    q_device = tq.QuantumDevice(n_wires=2)
    target_state = torch.tensor([0, 1, 0, 0], dtype=torch.complex64)

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
        train(target_state, q_device, model, optimizer)
        scheduler.step()


if __name__ == "__main__":
    main()
