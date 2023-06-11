"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
