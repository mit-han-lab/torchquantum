import torchquantum as tq

from torchquantum.algorithm import VQE, Hamiltonian
from qiskit import QuantumCircuit, transpile

from torchquantum.plugin import qiskit2tq_op_history, op_history2qiskit
from torchquantum.plugin import QiskitProcessor

from qiskit import IBMQ
IBMQ.load_account()

if __name__ == "__main__":
    hamil = Hamiltonian.from_file("./h2.txt")

    # or alternatively, you can use the following code to generate the ops
    circ = QuantumCircuit(2)
    circ.x(0)
    circ.rz(0.1, 1)
    circ.cx(0, 1)
    circ.x(0)
    circ.rz(0.1, 1)
    circ.cx(0, 1)

    ops = qiskit2tq_op_history(circ)
    print(ops)

    ansatz = tq.QuantumModule.from_op_history(ops)

    noise_model_tq = tq.NoiseModelTQ(
        noise_model_name="ibmq_quito",
    )

    noise_model_tq.v_c_reg_mapping = {'v2c': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
                                      'c2v': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
                                      }
    noise_model_tq.p_c_reg_mapping = {'p2c': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
                                      'c2p': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
                                      }
    noise_model_tq.p_v_reg_mapping ={'p2v': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
                                      'v2p': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6},
                                      }

    ansatz.set_noise_model_tq(noise_model_tq)

    configs = {
        "n_epochs": 10,
        "n_steps": 100,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "lr": 0.1,
        "device": "cuda",
    }
    
    vqe = VQE(
        hamil=hamil,
        ansatz=ansatz,
        train_configs=configs,
    )
    expval = vqe.train()
