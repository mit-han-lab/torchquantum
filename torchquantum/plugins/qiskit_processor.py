import torch
import torchquantum as tq

from qiskit import Aer, execute, IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from torchquantum.plugins import tq2qiskit
from torchquantum.utils import get_expectations_from_counts
from torchpack.utils.config import configs
from .qiskit_macros import IBMQ_NAMES
from tqdm import tqdm


class QiskitProcessor(object):
    def __init__(self,
                 use_real_qc=False,
                 backend_name=None,
                 noise_model_name=None,
                 coupling_map_name=None,
                 basis_gates_name=None,
                 n_shots=8192,
                 initial_layout=None,
                 seed_transpiler=42,
                 seed_simulator=42,
                 optimization_level=None,
                 ):
        self.use_real_qc = use_real_qc
        self.noise_model_name = noise_model_name
        self.backend_name = backend_name
        self.coupling_map_name = coupling_map_name
        self.basis_gates_name = basis_gates_name
        self.n_shots = n_shots
        self.initial_layout = initial_layout
        self.seed_transpiler = seed_transpiler
        self.seed_simulator = seed_simulator
        self.optimization_level = optimization_level

        self.backend = None
        self.provider = None
        self.noise_model = None
        self.coupling_map = None
        self.basis_gates = None

        self.qiskit_init()

    def get_noise_model(self, name):
        if name in IBMQ_NAMES:
            backend = self.provider.get_backend(name)
            noise_model = NoiseModel.from_backend(backend)
        else:
            noise_model = None

        return noise_model

    def get_coupling_map(self, name):
        if name in IBMQ_NAMES:
            backend = self.provider.get_backend(name)
            coupling_map = backend.configuration().coupling_map
        else:
            coupling_map = None

        return coupling_map

    def get_basis_gates(self, name):
        if name in IBMQ_NAMES:
            backend = self.provider.get_backend(name)
            basis_gates = backend.configuration().basis_gates
        else:
            basis_gates = None

        return basis_gates

    def qiskit_init(self):
        IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='ibm-q')

        if self.use_real_qc:
            self.backend = self.provider.get_backend(
                configs.qiskit.backend_name)
        else:
            # use simulator
            self.backend = Aer.get_backend('qasm_simulator')
            self.noise_model = self.get_noise_model(self.noise_model_name)
            self.coupling_map = self.get_coupling_map(self.coupling_map_name)
            self.basis_gates = self.get_basis_gates(self.basis_gates_name)

    def process(self, q_device: tq.QuantumDevice, q_layer: tq.QuantumModule,
                x):
        circs = []
        for i, x_single in tqdm(enumerate(x)):
            circ = tq2qiskit(q_layer, x_single.unsqueeze(0))
            circ.measure(list(range(q_device.n_wires)), list(range(
                q_device.n_wires)))
            circs.append(circ)

        job = execute(experiments=circs,
                      backend=self.backend,
                      shots=self.n_shots,
                      initial_layout=self.initial_layout,
                      seed_transpiler=self.seed_transpiler,
                      seed_simulator=self.seed_simulator,
                      coupling_map=self.coupling_map,
                      basis_gates=self.basis_gates,
                      noise_model=self.noise_model,
                      optimization_level=self.optimization_level,
                      )
        job_monitor(job, interval=1)

        result = job.result()
        counts = result.get_counts()

        measured_qiskit = get_expectations_from_counts(
            counts, n_wires=q_device.n_wires)
        measured_qiskit = torch.tensor(measured_qiskit, device=x.device)

        return measured_qiskit
