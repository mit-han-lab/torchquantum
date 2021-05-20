import torch
import torchquantum as tq
import pathos.multiprocessing as multiprocessing
import itertools

from qiskit import Aer, execute, IBMQ, transpile, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.monitor import job_monitor
from qiskit.exceptions import QiskitError
from torchquantum.plugins import tq2qiskit, tq2qiskit_parameterized
from torchquantum.utils import get_expectations_from_counts, get_provider
from .qiskit_macros import IBMQ_NAMES
from tqdm import tqdm
from torchpack.utils.logging import logger
from qiskit.transpiler import PassManager


class EmptyPassManager(PassManager):
    def run(
        self,
        circuits,
        output_name: str = None,
        callback=None
    ):
        return circuits


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
        except Exception as e:
            if "Job was cancelled" in str(e):
                logger.warning(f"Job is cancelled manually.")
                return None
            else:
                logger.warning(f"Job failed because {e}, rerun now.")

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
                 remove_ops=False,
                 remove_ops_thres=1e-4,
                 transpile_with_ancilla=True,
                 hub=None,
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
        self.transpile_with_ancilla = transpile_with_ancilla

        self.hub = hub
        self.backend = None
        self.provider = None
        self.noise_model = None
        self.coupling_map = None
        self.basis_gates = None
        self.properties = None
        self.empty_pass_manager = EmptyPassManager()

        self.transpiled_circs = None

        self.remove_ops = remove_ops
        self.remove_ops_thres = remove_ops_thres

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
        self.provider = get_provider(self.backend_name, hub=self.hub)

        if self.use_real_qc:
            self.backend = self.provider.get_backend(
                self.backend_name)
            self.properties = self.backend.properties()
            self.coupling_map = self.get_coupling_map(self.backend_name)
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
        if not self.transpile_with_ancilla and self.coupling_map is not None:
            # only use same number of physical qubits as virtual qubits
            # !! the risk is that the remaining graph is not a connected graph,
            # need fix this later
            coupling_map = []
            for pair in self.coupling_map:
                if all([p_wire < len(circs.qubits) for p_wire in pair]):
                    coupling_map.append(pair)
        else:
            coupling_map = self.coupling_map
        transpiled_circs = transpile(circuits=circs,
                                     backend=self.backend,
                                     basis_gates=self.basis_gates,
                                     coupling_map=coupling_map,
                                     initial_layout=self.initial_layout,
                                     seed_transpiler=self.seed_transpiler,
                                     optimization_level=self.optimization_level
                                     )
        return transpiled_circs

    def preprocess_parameterized(self,
                                 q_device,
                                 q_layer_parameterized,
                                 q_layer_fixed,
                                 q_layer_measure,
                                 x,
                                 ):
        circ_parameterized, params = tq2qiskit_parameterized(
            q_device, q_layer_parameterized.func_list)
        circ_fixed = tq2qiskit(q_device, q_layer_fixed,
                               remove_ops=self.remove_ops,
                               remove_ops_thres=self.remove_ops_thres)
        circ = circ_parameterized + circ_fixed

        v_c_reg_mapping = q_layer_measure.v_c_reg_mapping

        if v_c_reg_mapping is not None:
            for q_reg, c_reg in v_c_reg_mapping['v2c'].items():
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

        return transpiled_circ, binds_all

    def process_parameterized(self, q_device: tq.QuantumDevice,
                              q_layer_parameterized: tq.QuantumModule,
                              q_layer_fixed: tq.QuantumModule,
                              q_layer_measure: tq.QuantumModule,
                              x,
                              parallel=True):
        """
        separate the conversion, encoder part will be converted to a
        parameterized Qiskit QuantumCircuit. The remaining part will be a
        non-parameterized QuantumCircuit. In this case, only one time of
        compilation is required.

        q_layer_parameterized needs to have a func_list to specify the gates

        for parallel:
        JobManager has bugs when submitting job, so use multiprocessing instead
        """
        transpiled_circ, binds_all = self.preprocess_parameterized(
            q_device, q_layer_parameterized, q_layer_fixed,
            q_layer_measure, x)

        if parallel:
            if hasattr(self.backend.configuration(), 'max_experiments'):
                chunk_size = self.backend.configuration().max_experiments
            else:
                # using simulator, apply multithreading
                chunk_size = len(binds_all) // self.max_jobs

            split_binds = [binds_all[i:i + chunk_size] for i in range(
                0, len(binds_all), chunk_size)]

            qiskit_verbose = self.max_jobs <= 6
            feed_dicts = []
            for split_bind in split_binds:
                feed_dict = {
                    'experiments': transpiled_circ,
                    'backend': self.backend,
                    'pass_manager': self.empty_pass_manager,
                    'shots': self.n_shots,
                    'seed_simulator': self.seed_simulator,
                    'noise_model': self.noise_model,
                    'parameter_binds': split_bind,
                }
                feed_dicts.append([feed_dict, qiskit_verbose])

            p = multiprocessing.Pool(self.max_jobs)
            results = p.map(run_job_worker, feed_dicts)
            p.close()

            if all(isinstance(result, dict) for result in results):
                counts = results
            else:
                if isinstance(results[-1], dict):
                    results[-1] = [results[-1]]
                counts = list(itertools.chain(*results))
        else:
            job = execute(experiments=transpiled_circ,
                          backend=self.backend,
                          pass_manager=self.empty_pass_manager,
                          shots=self.n_shots,
                          seed_simulator=self.seed_simulator,
                          noise_model=self.noise_model,
                          parameter_binds=binds_all
                          )
            job_monitor(job, interval=1)

            result = job.result()
            counts = result.get_counts()

        measured_qiskit = get_expectations_from_counts(
            counts, n_wires=q_device.n_wires)
        measured_qiskit = torch.tensor(measured_qiskit, device=x.device)

        return measured_qiskit

    def process_multi_measure(self,
                              q_device: tq.QuantumDevice,
                              q_layer: tq.QuantumModule,
                              q_layer_measure: tq.QuantumModule,):
        obs_list = q_layer_measure.obs_list
        v_c_reg_mapping = q_layer_measure.v_c_reg_mapping
        circ_fixed = tq2qiskit(q_device, q_layer,
                               remove_ops=self.remove_ops,
                               remove_ops_thres=self.remove_ops_thres)

        transpiled_circ_fixed = self.transpile(circ_fixed)

        circ_all = []

        for hamil in obs_list:
            circ_diagonalize = QuantumCircuit(q_device.n_wires,
                                              q_device.n_wires)

            # diagonalize the measurements
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'x':
                    circ_diagonalize.h(qubit=wire)
                elif observable == 'y':
                    circ_diagonalize.z(qubit=wire)
                    circ_diagonalize.s(qubit=wire)
                    circ_diagonalize.h(qubit=wire)

            if v_c_reg_mapping is not None:
                for q_reg, c_reg in v_c_reg_mapping['v2c'].items():
                    circ_diagonalize.measure(q_reg, c_reg)
            else:
                circ_diagonalize.measure(list(range(q_device.n_wires)),
                                         list(range(q_device.n_wires)))

            transpiled_circ_diagonalize = self.transpile(circ_diagonalize)
            circ_all.append(transpiled_circ_fixed +
                            transpiled_circ_diagonalize)

        self.transpiled_circs = circ_all

        if hasattr(self.backend.configuration(), 'max_experiments'):
            chunk_size = self.backend.configuration().max_experiments
        else:
            # using simulator, apply multithreading
            chunk_size = len(circ_all) // self.max_jobs

        split_circs = [circ_all[i:i + chunk_size] for i in range(
            0, len(circ_all), chunk_size)]

        qiskit_verbose = self.max_jobs <= 2
        feed_dicts = []
        for split_circ in split_circs:
            feed_dict = {
                'experiments': split_circ,
                'backend': self.backend,
                'pass_manager': self.empty_pass_manager,
                'shots': self.n_shots,
                'seed_simulator': self.seed_simulator,
                'noise_model': self.noise_model,
            }
            feed_dicts.append([feed_dict, qiskit_verbose])

        p = multiprocessing.Pool(self.max_jobs)
        results = p.map(run_job_worker, feed_dicts)
        p.close()

        if all(isinstance(result, dict) for result in results):
            counts = results
        else:
            if isinstance(results[-1], dict):
                results[-1] = [results[-1]]
            counts = list(itertools.chain(*results))

        measured_qiskit = get_expectations_from_counts(
            counts, n_wires=q_device.n_wires)

        measured_qiskit = torch.tensor(measured_qiskit,
                                       device=q_device.state.device)

        return measured_qiskit

    def process(self, q_device: tq.QuantumDevice, q_layer: tq.QuantumModule,
                q_layer_measure: tq.QuantumModule, x):
        circs = []
        for i, x_single in tqdm(enumerate(x)):
            circ = tq2qiskit(q_device, q_layer, x_single.unsqueeze(0))
            if q_layer_measure.v_c_reg_mapping is not None:
                for q_reg, c_reg in q_layer_measure.v_c_reg_mapping[
                        'v2c'].items():
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
