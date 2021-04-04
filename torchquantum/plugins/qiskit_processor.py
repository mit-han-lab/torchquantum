import torch
import torchquantum as tq
import pathos.multiprocessing as multiprocessing
import itertools

from qiskit import Aer, execute, IBMQ, transpile
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from qiskit.exceptions import QiskitError
from torchquantum.plugins import tq2qiskit, tq2qiskit_parameterized
from torchquantum.utils import get_expectations_from_counts
from .qiskit_macros import IBMQ_NAMES
from tqdm import tqdm
# from qiskit.providers.ibmq.managed import IBMQJobManager
from torchpack.utils.logging import logger


def run_job_worker(data):
    while True:
        try:
            job = execute(**(data[0]))
            qiskit_verbose = data[1]
            if qiskit_verbose:
                job_monitor(job, interval=1)
            result = job.result()
            counts = result.get_counts()
            break
        except QiskitError:
            logger.warning('Job failed, rerun now.')

    return counts


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
                 max_jobs=5,
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
        self.max_jobs = max_jobs

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

    def process_parameterized_managed(
            self,
            q_device: tq.QuantumDevice,
            q_layer_parameterized: tq.QuantumModule,
            q_layer_fixed: tq.QuantumModule,
            x,
            q_c_reg_mapping=None):
        """
        with job manager, it will automatically separate the experiments,
        to use this, we have to explicitly expand circuits to make each binds
        only one set of parameter
        JobManager has bugs when submitting job, so use multiprocessing instead
        """
        # measured_qiskit_reference = self.process_parameterized(
        #     q_device, q_layer_parameterized, q_layer_fixed, x)

        circ_parameterized, params = tq2qiskit_parameterized(
            q_device, q_layer_parameterized.func_list)
        circ_fixed = tq2qiskit(q_device, q_layer_fixed)
        circ = circ_parameterized + circ_fixed

        if q_c_reg_mapping is not None:
            for q_reg, c_reg in q_c_reg_mapping['q2c'].items():
                circ.measure(q_reg, c_reg)
        else:
            circ.measure(list(range(q_device.n_wires)), list(range(
                q_device.n_wires)))

        transpiled_circ = self.transpile(circ)
        self.transpiled_circs = [transpiled_circ]

        # construct the circuits, each bind one set of parameters
        # bound_circs = []
        # for inputs_single in x:
        #     binds = {}
        #     for k, input_single in enumerate(inputs_single):
        #         binds[params[k]] = input_single.item()
        #     bound_circ = transpiled_circ.bind_parameters(binds)
        #     bound_circs.append(bound_circ)

        binds_all = []
        for inputs_single in x:
            binds = {}
            for k, input_single in enumerate(inputs_single):
                binds[params[k]] = input_single.item()
            binds_all.append(binds)

        # job_manager = IBMQJobManager()
        # job_set_foo = job_manager.run(experiments=bound_circs,
        #                               backend=self.backend,
        #                               name='foo',
        #                               seed_simulator=self.seed_simulator,
        #                               )
        #
        # results = job_set_foo.results()
        # counts = results.get_counts()
        if hasattr(self.backend.configuration(), 'max_experiments'):
            chunk_size = self.backend.configuration().max_experiments
        else:
            # using simulator, apply multithreading
            chunk_size = len(binds_all) // self.max_jobs

        split_binds = [binds_all[i:i + chunk_size] for i in range(0,
                       len(binds_all), chunk_size)]

        qiskit_verbose = self.max_jobs <= 6
        feed_dicts = []
        for split_bind in split_binds:
            feed_dict = {
                'experiments': transpiled_circ,
                'backend': self.backend,
                'shots': self.n_shots,
                'seed_transpiler': self.seed_transpiler,
                'seed_simulator': self.seed_simulator,
                'coupling_map': self.coupling_map,
                'basis_gates': self.basis_gates,
                'noise_model': self.noise_model,
                'optimization_level': self.optimization_level,
                'parameter_binds': split_bind,
            }
            feed_dicts.append([feed_dict, qiskit_verbose])

        p = multiprocessing.Pool(self.max_jobs)
        results = p.map(run_job_worker, feed_dicts)

        counts = list(itertools.chain(*results))
        measured_qiskit = get_expectations_from_counts(
            counts, n_wires=q_device.n_wires)
        p.close()

        measured_qiskit = torch.tensor(measured_qiskit, device=x.device)

        return measured_qiskit

    def process_parameterized(self, q_device: tq.QuantumDevice,
                              q_layer_parameterized: tq.QuantumModule,
                              q_layer_fixed: tq.QuantumModule,
                              x,
                              q_c_reg_mapping=None):
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

        if q_c_reg_mapping is not None:
            for q_reg, c_reg in q_c_reg_mapping['q2c'].items():
                circ.measure(q_reg, c_reg)
        else:
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
                x, q_c_reg_mapping=None):
        circs = []
        for i, x_single in tqdm(enumerate(x)):
            circ = tq2qiskit(q_device, q_layer, x_single.unsqueeze(0))
            if q_c_reg_mapping is not None:
                for q_reg, c_reg in q_c_reg_mapping['q2c'].items():
                    circ.measure(q_reg, c_reg)
            else:
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
