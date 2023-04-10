import torchquantum as tq
from torchquantum.measurement import expval_joint_analytical, expval_joint_sampling_grouping

import numpy as np
import random

def test_expval_joint_sampling_grouping():
    n_obs = 20
    n_wires = 4
    obs_all = []
    for _ in range(n_obs):
        obs = random.choices(["X", "Y", "Z", "I"], k=n_wires)
        obs_all.append("".join(obs))
    obs_all = list(set(obs_all))

    random_layer = tq.RandomLayer(n_ops=100, wires=list(range(n_wires)))
    qdev = tq.QuantumDevice(n_wires=n_wires, bsz=1, record_op=True)
    random_layer(qdev)

    expval_ana = {}
    for obs in obs_all:
        expval_ana[obs] = expval_joint_analytical(qdev, observable=obs)[0].item()
    
    expval_sam = expval_joint_sampling_grouping(qdev, observables=obs_all, n_shots_per_group=1000000)
    for obs in obs_all:
        # assert 
        assert np.isclose(expval_ana[obs], expval_sam[obs][0].item(), atol=1e-2)
        print(obs, expval_ana[obs], expval_sam[obs][0].item())

if __name__ == '__main__':
    test_expval_joint_sampling_grouping()
