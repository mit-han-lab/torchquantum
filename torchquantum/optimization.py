# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import argparse
#
# import torchquantum as tq
# import torchquantum.functional as tqf
#
# from torchquantum.plugins import (tq2qiskit_expand_params,
#                                   tq2qiskit,
#                                   tq2qiskit_measurement,
#                                   qiskit_assemble_circs,
#                                   op_history2qiskit,
#                                   op_history2qiskit_expand_params)
#
# from torchquantum.datasets import MNIST
# from torch.optim.lr_scheduler import CosineAnnealingLR
#
# import random
# import numpy as np
#
#
# class QFCModel(tq.QuantumModule):
#     class QLayer(tq.QuantumModule):
#         def __init__(self):
#             super().__init__()
#             self.n_wires = 4
#             self.random_layer = tq.RandomLayer(n_ops=50,
#                                                wires=list(range(self.n_wires)))
#
#             # gates with trainable parameters
#             self.rx0 = tq.RX(has_params=True, trainable=True)
#             self.ry0 = tq.RY(has_params=True, trainable=True)
#             self.rz0 = tq.RZ(has_params=True, trainable=True)
#             self.crx0 = tq.CRX(has_params=True, trainable=True)
#
#         @tq.static_support
#         def forward(self, qdev: tq.QuantumDevice):
#             """
#             1. To convert tq QuantumModule to qiskit or run in the static
#             model, need to:
#                 (1) add @tq.static_support before the forward
#                 (2) make sure to add
#                     static=self.static_mode and
#                     parent_graph=self.graph
#                     to all the tqf functions, such as tqf.hadamard below
#             """
#             self.random_layer(qdev)
#
#             # some trainable gates (instantiated ahead of time)
#             self.rx0(qdev, wires=0)
#             self.ry0(qdev, wires=1)
#             self.rz0(qdev, wires=3)
#             self.crx0(qdev, wires=[0, 2])
#
#             # add some more non-parameterized gates (add on-the-fly)
#             qdev.h(wires=3, static=self.static_mode, parent_graph=self.graph)
#             qdev.sx(wires=2, static=self.static_mode, parent_graph=self.graph)
#             qdev.cnot(wires=[3, 0], static=self.static_mode, parent_graph=self.graph)
#             qdev.rx(wires=1, params=torch.tensor([0.1]), static=self.static_mode, parent_graph=self.graph)
#
#     def __init__(self):
#         super().__init__()
#         self.n_wires = 4
#         self.encoder = tq.GeneralEncoder(
#             tq.encoder_op_list_name_dict['4x4_u3rx'])
#
#         self.q_layer = self.QLayer()
#         self.measure = tq.MeasureAll(tq.PauliZ)
#
#     def forward(self, x, use_qiskit=False):
#         qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True)
#
#         bsz = x.shape[0]
#         x = F.avg_pool2d(x, 6).view(bsz, 16)
#         devi = x.device
#
#         if use_qiskit:
#             encoder_circs = tq2qiskit_expand_params(qdev, x,
#                                                     self.encoder.func_list)
#             q_layer_circ = tq2qiskit(qdev, self.q_layer)
#             measurement_circ = tq2qiskit_measurement(qdev,
#                                                      self.measure)
#             assembled_circs = qiskit_assemble_circs(encoder_circs,
#                                                     q_layer_circ,
#                                                     measurement_circ)
#             x0 = self.qiskit_processor.process_ready_circs(
#                 qdev, assembled_circs).to(devi)
#             x = x0
#
#         else:
#             self.encoder(qdev, x)
#             op_history_parameterized = qdev.op_history
#             qdev.reset_op_history()
#             self.q_layer(qdev)
#             op_history_fixed = qdev.op_history
#             x = self.measure(qdev)
#
#         # circs = op_history2qiskit_expand_params(self.n_wires, op_history_parameterized, bsz=bsz)
#         # print(op_history2qiskit(self.n_wires, op_history_fixed))
#
#         x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
#         x = F.log_softmax(x, dim=1)
#
#         return x
#
#
# def train(dataflow, model, device, optimizer):
#     for feed_dict in dataflow['train']:
#         inputs = feed_dict['image'].to(device)
#         targets = feed_dict['digit'].to(device)
#
#         outputs = model(inputs)
#         loss = F.nll_loss(outputs, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f"loss: {loss.item()}", end='\r')
#
#
# def valid_test(dataflow, split, model, device, qiskit=False):
#     target_all = []
#     output_all = []
#     with torch.no_grad():
#         for feed_dict in dataflow[split]:
#             inputs = feed_dict['image'].to(device)
#             targets = feed_dict['digit'].to(device)
#
#             outputs = model(inputs, use_qiskit=qiskit)
#
#             target_all.append(targets)
#             output_all.append(outputs)
#         target_all = torch.cat(target_all, dim=0)
#         output_all = torch.cat(output_all, dim=0)
#
#     _, indices = output_all.topk(1, dim=1)
#     masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
#     size = target_all.shape[0]
#     corrects = masks.sum().item()
#     accuracy = corrects / size
#     loss = F.nll_loss(output_all, target_all).item()
#
#     print(f"{split} set accuracy: {accuracy}")
#     print(f"{split} set loss: {loss}")
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--static', action='store_true', help='compute with '
#                                                               'static mode')
#     parser.add_argument('--pdb', action='store_true', help='debug with pdb')
#     parser.add_argument('--wires-per-block', type=int, default=2,
#                         help='wires per block int static mode')
#     parser.add_argument('--epochs', type=int, default=5,
#                         help='number of training epochs')
#
#     args = parser.parse_args()
#
#     if args.pdb:
#         import pdb
#         pdb.set_trace()
#
#     seed = 0
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     dataset = MNIST(
#         root='./mnist_data',
#         train_valid_split_ratio=[0.9, 0.1],
#         digits_of_interest=[3, 6],
#         n_test_samples=75,
#     )
#     dataflow = dict()
#
#     for split in dataset:
#         sampler = torch.utils.data.RandomSampler(dataset[split])
#         dataflow[split] = torch.utils.data.DataLoader(
#             dataset[split],
#             batch_size=256,
#             sampler=sampler,
#             num_workers=8,
#             pin_memory=True)
#
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     model = QFCModel().to(device)
#
#     n_epochs = args.epochs
#     optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
#     scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
#
#     if args.static:
#         # optionally to switch to the static mode, which can bring speedup
#         # on training
#         model.q_layer.static_on(wires_per_block=args.wires_per_block)
#
#     for epoch in range(1, n_epochs + 1):
#         # train
#         print(f"Epoch {epoch}:")
#         train(dataflow, model, device, optimizer)
#         print(optimizer.param_groups[0]['lr'])
#
#         # valid
#         valid_test(dataflow, 'valid', model, device)
#         scheduler.step()
#
#     # test
#     valid_test(dataflow, 'test', model, device, qiskit=False)
#
#     # run on Qiskit simulator and real Quantum Computers
#     try:
#         from qiskit import IBMQ
#         from torchquantum.plugins import QiskitProcessor
#
#         # firstly perform simulate
#         print(f"\nTest with Qiskit Simulator")
#         processor_simulation = QiskitProcessor(use_real_qc=False)
#         model.set_qiskit_processor(processor_simulation)
#         valid_test(dataflow, 'test', model, device, qiskit=True)
#
#         # then try to run on REAL QC
#         backend_name = 'ibmq_lima'
#         print(f"\nTest on Real Quantum Computer {backend_name}")
#         # Please specify your own hub group and project if you have the
#         # IBMQ premium plan to access more machines.
#         processor_real_qc = QiskitProcessor(use_real_qc=True,
#                                             backend_name=backend_name,
#                                             hub='ibm-q',
#                                             group='open',
#                                             project='main',
#                                             )
#         model.set_qiskit_processor(processor_real_qc)
#         valid_test(dataflow, 'test', model, device, qiskit=True)
#     except ImportError:
#         print("Please install qiskit, create an IBM Q Experience Account and "
#               "save the account token according to the instruction at "
#               "'https://github.com/Qiskit/qiskit-ibmq-provider', "
#               "then try again.")
#
#
# if __name__ == '__main__':
#     main()

import numpy as np
import time
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from scipy.optimize import LinearConstraint
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from scipy.stats import norm


def acquisition(x_scaled, hyper_param, model, min_Y):  # x_scaled: 1 * dim
    x_scaled = x_scaled.reshape(1, -1)
    if "LCB" in hyper_param[0]:
        mean, std = model.predict(x_scaled, return_std=True)
        return mean[0] - hyper_param[1] * std[0]
    elif hyper_param[0] == "EI":
        tau = min_Y
        mean, std = model.predict(x_scaled, return_std=True)
        tau_scaled = (tau - mean) / std
        res = (tau - mean) * norm.cdf(tau_scaled) + std * norm.pdf(tau_scaled)
        return -res  # maximize Ei = minimize -EI
    elif hyper_param[0] == "PI":
        tau = min_Y
        mean, std = model.predict(x_scaled, return_std=True)
        tau_scaled = (tau - mean) / std
        res = norm.cdf(tau_scaled)
        return -res
    else:
        raise ValueError("acquisition function is not implemented")


def bayes_opt(
    func,
    dim_design,
    N_sim,
    N_initial,
    w_bound,
    hyper_param,
    store=False,
    verbose=True,
    file_suffix="",
):
    """

    :param func: [functional handle], represents the objective function. objective = func(design)
    :param dim_design: [int], the dimension of the design variable
    :param N_sim: [int], The total number of allowable simulations
    :param N_initial: [int], The number of simulations used to set up the initial dataset
    :param w_bound: [(dim_design, 2) np.array], the i-th row contains the lower bound and upper bound for the i-th variable
    :param hyper_param: the parameter for the acquisition function e.g., ['LCB','0.3'], ['EI'], ['PI']
    :param verbose: [Bool], if it is true, print detailed information in each iteration of Bayesian optimization
    :param file_suffix: [string], file suffix used in storing optimization information
    :return:
    cur_best_w: [(dim_design,) np.array], the best design variable
    cur_best_y: [float], the minimum objective value
    """

    # initialization: set up the training dataset X, Y.
    print("Begin initializing...")
    X = np.repeat(
        (w_bound[:, 1] - w_bound[:, 0]).reshape(1, -1), N_initial, axis=0
    ) * np.random.rand(N_initial, dim_design) + np.repeat(
        w_bound[:, 0].reshape(1, -1), N_initial, axis=0
    )
    Y = np.zeros((N_initial,))

    for i in range(N_initial):
        Y[i] = func(X[i, :])
        print(
            "Simulate the %d-th sample... with metric: %.3e" % (i, Y[i])
        ) if verbose else None
    print("Finish initialization with best metric: %.3e" % (np.min(Y)))

    # define several working variables, will be used to store results
    pred_mean = np.zeros(N_sim - N_initial)
    pred_std = np.zeros(N_sim - N_initial)
    acq_list = np.zeros(N_sim - N_initial)

    # Goes into real Bayesian Optimization
    cur_count, cur_best_w, cur_best_y = N_initial, None, 1e10
    while cur_count < N_sim:

        # build gaussian process on the normalized data
        wrk_mean, wrk_std = X.mean(axis=0), X.std(axis=0)
        model = GPR(
            kernel=ConstantKernel(1, (1e-9, 1e9)) * RBF(1.0, (1e-5, 1e5)),
            normalize_y=True,
            n_restarts_optimizer=100,
        )
        model.fit(np.divide(X - wrk_mean, wrk_std), Y)

        # define acquisition function, np.min(Y) is needed in EI and PI, but not LCB
        acq_func = lambda x_scaled: acquisition(x_scaled, hyper_param, model, np.min(Y))

        # optimize the acquisition function independently for N_inner times, select the best one
        N_inner, cur_min, opt = 20, np.inf, None
        for i in range(N_inner):
            w_init = (w_bound[:, 1] - w_bound[:, 0]) * np.random.rand(dim_design) + (
                w_bound[:, 0]
            )
            LC = LinearConstraint(
                np.eye(dim_design),
                np.divide(w_bound[:, 0] - wrk_mean, wrk_std),
                np.divide(w_bound[:, 1] - wrk_mean, wrk_std),
                keep_feasible=False,
            )
            cur_opt = minimize(
                acq_func,
                np.divide(w_init - wrk_mean, wrk_std),
                method="COBYLA",
                constraints=LC,
                options={"disp": False},
            )
            wrk = acq_func(cur_opt.x)
            if cur_min >= wrk:
                cur_min = wrk
                opt = cur_opt

        # do a clipping to avoid violation of constraints (just in case), and also undo the normalization
        newX = np.clip(opt.x * wrk_std + wrk_mean, w_bound[:, 0], w_bound[:, 1])
        star_time = time.time()
        cur_count += 1
        newY = func(newX)
        end_time = time.time()
        X, Y = np.concatenate((X, newX.reshape(1, -1)), axis=0), np.concatenate(
            (Y, [newY]), axis=0
        )

        # save and display information
        ind = np.argmin(Y)
        cur_predmean, cur_predstd = model.predict(
            (np.divide(newX - wrk_mean, wrk_std)).reshape(1, -1), return_std=True
        )
        cur_acq = acq_func(np.divide(newX - wrk_mean, wrk_std))
        cur_best_w, cur_best_y = X[ind, :], Y[ind]
        pred_mean[cur_count - N_initial - 1], pred_std[cur_count - N_initial - 1] = (
            cur_predmean,
            cur_predstd,
        )
        acq_list[cur_count - N_initial - 1] = cur_acq
        if store:
            np.save("./result/X_" + file_suffix + ".npy", X)
            np.save("./result/Y_" + file_suffix + ".npy", Y)
            np.save("./result/cur_best_w_" + file_suffix + ".npy", cur_best_w)
            np.save("./result/cur_best_y_" + file_suffix + ".npy", cur_best_y)
            np.save("./result/pred_mean_" + file_suffix + ".npy", pred_mean)
            np.save("./result/pred_std_" + file_suffix + ".npy", pred_std)
            np.save("./result/acq_list_" + file_suffix + ".npy", acq_list)
        if verbose:
            print("-" * 10)
            print("Number of function evaluations: %d" % cur_count)
            print("Optimize acq message: ", opt.message)
            print(
                "Model predict(new sampled X)... mean: %.3e, std:%.3e"
                % (cur_predmean, cur_predstd)
            )
            print("Acq(new sampled X): %.3e" % cur_acq)
            print(
                "Y(new sampled X): %.3e, simulation time: %.3e"
                % (newY, end_time - star_time)
            )
            print("Current best design: ", cur_best_w)
            print("Current best function value: %.3e" % cur_best_y)

    return cur_best_w, cur_best_y


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    # example: minimize x1^2 + x2^2 + x3^2 + ...
    dim_design = 10
    N_total = 200
    N_initial = 40
    bound = np.ones((dim_design, 2)) * np.array([-10, 10])  # -inf < xi < inf

    func = lambda x: np.sum(x * x)
    cur_best_w, cur_best_y = bayes_opt(
        func,
        dim_design,
        N_total,
        N_initial,
        bound,
        ["LCB", 0.3],
        store=False,
        verbose=True,
        file_suffix=str(seed),
    )

    print(cur_best_w)
    print(cur_best_y)
