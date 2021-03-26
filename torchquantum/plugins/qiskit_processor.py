import torch
import torchquantum as tq

from qiskit import Aer, execute, IBMQ
from qiskit.compiler import transpile
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from torchquantum.plugins import tq2qiskit, tq2qiskit_parameterized
from torchquantum.utils import get_expectations_from_counts
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
        self.properties = None

        self.transpiled_circs = None

        self.qiskit_init()

    def get_noise_model(self, name):
        if name in IBMQ_NAMES:
            backend = self.provider.get_backend(name)
            self.properties = backend.properties()
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
        self.backend = None
        self.provider = None
        self.noise_model = None
        self.coupling_map = None
        self.basis_gates = None
        self.properties = None

        IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='ibm-q')

        if self.use_real_qc:
            self.backend = self.provider.get_backend(
                self.backend_name)
            self.properties = self.backend.properties()
        else:
            # use simulator
            self.backend = Aer.get_backend('qasm_simulator',
                                           max_parallel_experiments=0)
            self.noise_model = self.get_noise_model(self.noise_model_name)
            self.coupling_map = self.get_coupling_map(self.coupling_map_name)
            self.basis_gates = self.get_basis_gates(self.basis_gates_name)

    def set_layout(self, layout):
        self.initial_layout = layout

    def transpile(self, circs):
        transpiled_circs = transpile(circuits=circs,
                                     backend=self.backend,
                                     basis_gates=self.basis_gates,
                                     coupling_map=self.coupling_map,
                                     initial_layout=self.initial_layout,
                                     seed_transpiler=self.seed_transpiler,
                                     optimization_level=self.optimization_level
                                     )
        return transpiled_circs

    def process_parameterized(self, q_device: tq.QuantumDevice,
                              q_layer_parameterized: tq.QuantumModule,
                              q_layer_fixed: tq.QuantumModule,
                              x):
        """
        separate the conversion, encoder part will be converted to a
        parameterized Qiskit QuantumCircuit. The remaining part will be a
        non-parameterized QuantumCircuit. In this case, only one time of
        compilation is required.

        q_layer_parameterized needs to have a func_list to specify the gates
        """
        circ_parameterized, params = tq2qiskit_parameterized(
            q_device, q_layer_parameterized.func_list)
        circ_fixed = tq2qiskit(q_device, q_layer_fixed)
        circ = circ_parameterized + circ_fixed
        circ.measure(list(range(q_device.n_wires)), list(range(
            q_device.n_wires)))

        transpiled_circ = self.transpile(circ)
        self.transpiled_circs = [transpiled_circ]
        # construct the parameter_binds
        binds_all = []
        for inputs_single in x:
            binds = {}
            for k, input_single in enumerate(inputs_single):
                binds[params[k]] = input_single.item()
            binds_all.append(binds)

        job = execute(experiments=transpiled_circ,
                      backend=self.backend,
                      shots=self.n_shots,
                      seed_transpiler=self.seed_transpiler,
                      seed_simulator=self.seed_simulator,
                      coupling_map=self.coupling_map,
                      basis_gates=self.basis_gates,
                      noise_model=self.noise_model,
                      optimization_level=self.optimization_level,
                      parameter_binds=binds_all
                      )
        job_monitor(job, interval=1)

        result = job.result()
        counts = result.get_counts()

        measured_qiskit = get_expectations_from_counts(
            counts, n_wires=q_device.n_wires)
        measured_qiskit = torch.tensor(measured_qiskit, device=x.device)

        return measured_qiskit

    def process(self, q_device: tq.QuantumDevice, q_layer: tq.QuantumModule,
                x):
        circs = []
        for i, x_single in tqdm(enumerate(x)):
            circ = tq2qiskit(q_device, q_layer, x_single.unsqueeze(0))
            circ.measure(list(range(q_device.n_wires)), list(range(
                q_device.n_wires)))
            circs.append(circ)

        transpiled_circs = self.transpile(circs)
        self.transpiled_circs = transpiled_circs

        job = execute(experiments=transpiled_circs,
                      backend=self.backend,
                      shots=self.n_shots,
                      # initial_layout=self.initial_layout,
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
