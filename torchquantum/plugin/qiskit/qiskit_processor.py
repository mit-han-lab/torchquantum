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

import torch
import torchquantum as tq
from concurrent.futures import ThreadPoolExecutor
import itertools
import warnings # Added for handling deprecation warnings

from qiskit import transpile, QuantumCircuit
# Removed: from qiskit import execute
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
# Removed: from .my_job_monitor import my_job_monitor as job_monitor
# Removed: from qiskit.providers.ibmq import IBMQ
from qiskit_ibm_runtime import QiskitRuntimeService # Changed provider to runtime
# from qiskit_aer.primitives import SamplerV2 as AerSamplerV2 # Added
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler # Changed provider to runtime
from qiskit.primitives.containers import PubResult # Added
from qiskit.exceptions import QiskitError
from .qiskit_plugin import (
    tq2qiskit,
    tq2qiskit_parameterized,
    tq2qiskit_measurement,
)
from torchquantum.util import (
    get_expectations_from_counts,
    # Removed: get_provider (IBMQ specific)
    # Removed: get_provider_hub_group_project (IBMQ specific)
    get_circ_stats,
)
from .qiskit_macros import IBMQ_NAMES # Keep for checking names? Or remove? Let's keep for now.
from tqdm import tqdm
from torchpack.utils.logging import logger
from qiskit.transpiler import PassManager
import numpy as np
import datetime


class EmptyPassManager(PassManager):
    def run(self, circuits, output_name: str = None, callback=None):
        return circuits

# Reworked worker function for SamplerV2
def run_job_worker_v2(job_data, max_retries=5):
    sampler_instance, pubs, run_options = job_data
    result = None # Initialize result
    for attempt in range(max_retries):
        try:
            # Use SamplerV2 run method
            job = sampler_instance.run(pubs, **run_options)
            result = job.result() # SamplerV2 returns PrimitiveResult directly
            # logger.info(f'SamplerV2 job successful, number of pubs: {len(pubs)}')
            break
        except Exception as e:
            # Handle potential errors like cancellation or other job failures
            if "Job was cancelled" in str(e) or "cancelled" in str(e).lower():
                logger.warning(f"Job was cancelled manually or by the system.")
                return None # Indicate cancellation
            else:
                logger.warning(
                    f"Sampler job failed because {e}, "
                    f"retry {attempt + 1}/{max_retries}."
                )
                import time
                time.sleep(1)

    if result is None:
        return None

    # Extract counts from result
    counts_list = []
    for pub_result in result:
        try:
            # SamplerV2 stores results in pub_result.data.<output_name>
            # Default output name for measurements is often 'meas' or the classical register name (e.g., 'c')
            # Check available keys if unsure
            data_keys = list(pub_result.data.keys())
            data_container = None
            if 'meas' in data_keys: # Prioritize 'meas' if present
                 data_container = pub_result.data['meas']
            elif 'c' in data_keys: # Try 'c' as common classical register name
                 data_container = pub_result.data['c']
            elif data_keys: # Fallback to the first key if others not found
                 data_container = pub_result.data[data_keys[0]]
                 logger.warning(f"Using fallback data key '{data_keys[0]}' for counts extraction.")
            else:
                 raise ValueError("No data keys found in PubResult to extract counts from.")

            # The container should have get_counts()
            counts_dict = data_container.get_counts()
            counts_list.append(counts_dict)
        except (KeyError, AttributeError, ValueError) as e:
             logger.error(f"Error extracting counts from PubResult: {e}. PubResult keys: {list(pub_result.data.keys())}")
             counts_list.append(None) # Append None if extraction failed
        except Exception as e:
            logger.error(f"Unexpected error extracting counts from PubResult: {e}")
            counts_list.append(None) # Append None for other errors

    return counts_list # Return list of counts dicts or Nones


class QiskitProcessor(object):
    def __init__(
        self,
        use_real_qc=False,
        backend_name=None,
        backend=None,
        noise_model_name=None,
        noise_model=None,
        coupling_map=None,
        basis_gates=None,
        n_shots=8192,
        initial_layout=None,
        seed_transpiler=42,
        seed_simulator=42,
        optimization_level=1,
        max_jobs=5,
        remove_ops=False,
        remove_ops_thres=1e-4,
        transpile_with_ancilla=True,
        ibm_quantum_token=None,
        layout_method=None,
        routing_method=None,
    ):
        self.use_real_qc = use_real_qc
        self.backend_name = backend_name
        self.noise_model_name = noise_model_name
        self.n_shots = n_shots
        self.initial_layout = initial_layout
        self.seed_transpiler = seed_transpiler
        self.seed_simulator = seed_simulator
        self.optimization_level = optimization_level
        self.max_jobs = max_jobs
        self.transpile_with_ancilla = transpile_with_ancilla

        self.layout_method = layout_method
        self.routing_method = routing_method

        self.ibm_quantum_token = ibm_quantum_token
        self.backend = backend
        self.service = None
        self.sampler = None
        self.noise_model = noise_model
        self.coupling_map = coupling_map
        self.basis_gates = basis_gates
        self.properties = None
        self.empty_pass_manager = EmptyPassManager()

        self.transpiled_circs = None

        self.remove_ops = remove_ops
        self.remove_ops_thres = remove_ops_thres

        self.qiskit_init()

    def qiskit_init(self):
        self.service = None
        self.sampler = None
        self.backend = None

        if self.use_real_qc:
            if self.backend_name is None:
                raise ValueError("backend_name must be provided if use_real_qc is True")
            try:
                self.service = QiskitRuntimeService(token=self.ibm_quantum_token, channel='ibm_quantum_platform')
                self.backend = self.service.backend(self.backend_name)
                self.sampler = RuntimeSampler(mode=self.backend)
                logger.info(f"Initialized QiskitRuntimeService and RuntimeSampler for backend: {self.backend_name}")
            except Exception as e:
                logger.error(f"Failed to initialize QiskitRuntimeService or get backend: {e}")
                raise
            if self.coupling_map is None:
                 self.coupling_map = self.backend.coupling_map
            if self.basis_gates is None:
                 self.basis_gates = getattr(
                     self.backend, "basis_gates", list(self.backend.target.operation_names)
                 )

        else:
            if self.noise_model is None and self.noise_model_name is not None:
                logger.info(f"Fetching noise model for backend: {self.noise_model_name}")
                try:
                    if self.ibm_quantum_token:
                        temp_service = QiskitRuntimeService(token=self.ibm_quantum_token, channel='ibm_quantum_platform')
                        temp_backend = temp_service.backend(self.noise_model_name)
                        self.noise_model = NoiseModel.from_backend(temp_backend)
                        logger.info(f"Successfully fetched noise model for {self.noise_model_name}")
                        if self.coupling_map is None:
                            self.coupling_map = temp_backend.coupling_map
                        if self.basis_gates is None:
                            self.basis_gates = getattr(
                                temp_backend, "basis_gates", list(temp_backend.target.operation_names)
                            )
                    else:
                        logger.warning("IBM Quantum token needed to fetch noise model by name, but not provided. Proceeding without noise model.")
                        self.noise_model = None
                except Exception as e:
                    logger.warning(f"Could not fetch noise model for {self.noise_model_name}: {e}. Proceeding without noise model.")
                    self.noise_model = None
            elif self.noise_model is not None:
                 logger.info("Using user-provided noise model instance.")
            else:
                 logger.info("No noise model specified or fetched.")
                 self.noise_model = None

            # Create AerSimulator backend (needed for transpilation)
            self.backend = AerSimulator(noise_model=self.noise_model)
            # Configure backend options for the sampler
            backend_opts = {"noise_model": self.noise_model} if self.noise_model else {}
            # Initialize Sampler with options
            self.sampler = AerSampler(options={"backend_options": backend_opts}, seed=self.seed_simulator)
            logger.info(f"Initialized AerSampler.{' With noise model.' if self.noise_model else ''}")

    def set_layout(self, layout):
        self.initial_layout = layout

    def set_backend(self, backend):
        logger.warning("Setting backend directly. Consider re-initializing QiskitProcessor for consistency.")
        self.backend = backend

    def transpile(self, circs):
        if isinstance(circs, QuantumCircuit):
            circs = [circs]

        if self.backend is None:
             logger.warning("No backend available for transpilation. Skipping.")
             return circs

        transpile_options = {
            "backend": self.backend,
            "optimization_level": self.optimization_level,
            "seed_transpiler": self.seed_transpiler,
            "layout_method": self.layout_method,
            "routing_method": self.routing_method,
            "initial_layout": self.initial_layout,
            **({"coupling_map": self.coupling_map} if self.coupling_map else {}),
            **({"basis_gates": self.basis_gates} if self.basis_gates else {}),
        }

        try:
            transpiled_circs = transpile(circuits=circs, **transpile_options)
        except Exception as e:
            logger.error(f"Transpilation failed: {e}")
            raise
        return transpiled_circs

    def _run_pubs_get_counts(self, pubs, parallel=True):
        """Run SamplerV2 pubs (optionally carrying parameter bindings) and
        return one counts dict (or None on failure) per pub."""
        if self.sampler is None:
            raise RuntimeError("QiskitProcessor not initialized. Call qiskit_init() first.")

        expected_pubs = len(pubs)

        # Prepare run options
        run_options = {"shots": self.n_shots}
        if isinstance(self.sampler, AerSampler):
            # Pass seed to constructor, not run options
            # run_options["seed"] = self.seed_simulator # Incorrect - seed is for constructor
            pass # Seed already set in constructor

        all_counts = []

        if parallel and len(pubs) > 1:
            # Determine chunk size for parallel processing
            num_pubs = len(pubs)
            # Adjust chunk_size calculation to avoid zero chunks if num_pubs < max_jobs
            chunk_size = (num_pubs + self.max_jobs - 1) // self.max_jobs if self.max_jobs > 0 else num_pubs
            if chunk_size == 0: chunk_size = 1 # Ensure chunk_size is at least 1

            split_pubs = [
                pubs[i : i + chunk_size] for i in range(0, num_pubs, chunk_size)
            ]
            logger.info(f"Processing {num_pubs} pubs in {len(split_pubs)} chunks using {self.max_jobs} workers.")

            job_data_list = [(self.sampler, pub_batch, run_options) for pub_batch in split_pubs]

            # Threads instead of forked processes: Aer's OpenMP thread pool is
            # not fork-safe (a fork after any prior job deadlocks the child),
            # Aer releases the GIL during simulation, and runtime jobs wait
            # server-side, so threads give the same concurrency safely.
            with ThreadPoolExecutor(max_workers=self.max_jobs) as pool:
                # results is now a list of lists (or Nones)
                batch_results = list(pool.map(run_job_worker_v2, job_data_list))

            # Process results: flatten the list of lists
            processed_pubs_count = 0
            for batch_counts_list in batch_results:
                if batch_counts_list is None:
                    # Need to know how many pubs were in the failed batch
                    # For simplicity, just log warning - length check later will catch discrepancy
                    logger.warning("A worker job batch failed or was cancelled. Results for this batch are lost.")
                elif isinstance(batch_counts_list, list):
                     all_counts.extend(batch_counts_list) # Extend with the list of counts/Nones from the worker
                     processed_pubs_count += len(batch_counts_list)
                else:
                     logger.warning(f"Unexpected item in results list: {type(batch_counts_list)}")

            if processed_pubs_count != expected_pubs:
                 logger.warning(f"Expected {expected_pubs} results, but only processed {processed_pubs_count} due to potential batch failures.")

        else: # Process sequentially
            logger.info(f"Processing {expected_pubs} pubs sequentially.")
            try:
                # run_job_worker_v2 now returns the list of counts/Nones directly
                all_counts = run_job_worker_v2((self.sampler, pubs, run_options))
                if all_counts is None: # Check if the sequential run itself failed
                    logger.error("Sequential job failed or was cancelled.")
                    all_counts = [None] * expected_pubs # Mark all as failed if job cancelled

            except Exception as e:
                 logger.error(f"Sequential SamplerV2 run failed: {e}")
                 all_counts = [None] * expected_pubs # Mark all as failed

        # Final check on length, although parallel processing makes exact padding difficult without more info
        if len(all_counts) != expected_pubs:
             logger.warning(f"Final number of results ({len(all_counts)}) does not match number of input circuits ({expected_pubs}). Results might be incomplete due to errors.")

        return all_counts # Return list of counts dictionaries or Nones

    def process_ready_circs_get_counts(self, circs_all, parallel=True):
        if self.sampler is None:
            raise RuntimeError("QiskitProcessor not initialized. Call qiskit_init() first.")

        # Transpile circuits
        logger.info(f"Transpiling {len(circs_all)} circuits...")
        # Ensure circs_all is a list
        if not isinstance(circs_all, list):
             circs_all = [circs_all]
        transpiled_circs = self.transpile(circs_all)
        logger.info("Transpilation complete.")

        # Package circuits into PUBS (Primitive Unified Blocs) for SamplerV2
        # Each pub is just the circuit for basic sampling
        pubs = [(circ,) for circ in transpiled_circs]
        return self._run_pubs_get_counts(pubs, parallel=parallel)

    def process_ready_circs(self, q_device, circs_all, parallel=True):
        counts_list = self.process_ready_circs_get_counts(circs_all, parallel=parallel)
        valid_counts = [counts for counts in counts_list if counts is not None]
        if len(valid_counts) != len(counts_list):
             logger.warning("Some circuits failed execution. Expectation values will only be calculated for successful runs.")

        if not valid_counts:
            logger.error("No circuits executed successfully.")
            return torch.empty(0, dtype=torch.float)

        measured_qiskit = get_expectations_from_counts(valid_counts, n_wires=q_device.n_wires)
        measured_torch = torch.tensor(measured_qiskit, dtype=torch.float)

        return measured_torch

    def preprocess_parameterized(
        self,
        q_device,
        q_layer_parameterized,
        q_layer_fixed,
        q_layer_measure,
        x,
    ):
        circ_parameterized, params = tq2qiskit_parameterized(
            q_device, q_layer_parameterized.func_list
        )
        circ_fixed = tq2qiskit(
            q_device,
            q_layer_fixed,
            remove_ops=self.remove_ops,
            remove_ops_thres=self.remove_ops_thres,
        )
        circ_measurement = tq2qiskit_measurement(q_device, q_layer_measure)

        circ = QuantumCircuit(q_device.n_wires, q_device.n_wires)
        circ.compose(circ_parameterized, inplace=True)
        circ.compose(circ_fixed, inplace=True)
        circ.compose(circ_measurement, inplace=True)

        logger.info(f"Before transpile: {get_circ_stats(circ)}")
        transpiled_circ = self.transpile(circ)[0]
        logger.info(f"After transpile: {get_circ_stats(transpiled_circ)}")
        self.transpiled_circs = [transpiled_circ]
        # construct the parameter binds, one dict per input sample
        binds_all = []
        for inputs_single in x:
            binds = {}
            for k, input_single in enumerate(inputs_single):
                binds[params[k]] = input_single.item()
            binds_all.append(binds)

        return transpiled_circ, binds_all

    def process_parameterized(
        self,
        q_device: tq.QuantumDevice,
        q_layer_parameterized: tq.QuantumModule,
        q_layer_fixed: tq.QuantumModule,
        q_layer_measure: tq.QuantumModule,
        x,
        parallel=True,
    ):
        """
        separate the conversion, encoder part will be converted to a
        parameterized Qiskit QuantumCircuit. The remaining part will be a
        non-parameterized QuantumCircuit. In this case, only one time of
        compilation is required. Each parameter binding becomes one
        SamplerV2 pub sharing the same transpiled circuit.
        """
        transpiled_circ, binds_all = self.preprocess_parameterized(
            q_device, q_layer_parameterized, q_layer_fixed, q_layer_measure, x
        )

        pubs = [(transpiled_circ, binds) for binds in binds_all]
        counts = self._run_pubs_get_counts(pubs, parallel=parallel)

        measured_qiskit = get_expectations_from_counts(counts, n_wires=q_device.n_wires)
        measured_qiskit = torch.tensor(measured_qiskit, dtype=torch.float, device=x.device)

        return measured_qiskit

    def preprocess_parameterized_and_shift(
        self,
        q_device,
        q_layer_parameterized,
        q_layer_fixed,
        q_layer_measure,
        x,
        shift_encoder,
        shift_this_step,
    ):
        circ_parameterized, params = tq2qiskit_parameterized(
            q_device, q_layer_parameterized.func_list
        )
        circ_fixed_list = []
        circ_fixed = tq2qiskit(
            q_device,
            q_layer_fixed,
            remove_ops=self.remove_ops,
            remove_ops_thres=self.remove_ops_thres,
        )
        circ_fixed_list.append(circ_fixed)

        # not shift encoder ==> shift fixed layer
        if not shift_encoder:
            for i, named_param in enumerate(q_layer_fixed.named_parameters()):
                if shift_this_step[i]:
                    param = named_param[-1]
                    param.copy_(param + np.pi * 0.5)
                    circ_fixed = tq2qiskit(
                        q_device,
                        q_layer_fixed,
                        remove_ops=self.remove_ops,
                        remove_ops_thres=self.remove_ops_thres,
                    )
                    circ_fixed_list.append(circ_fixed)
                    param.copy_(param - np.pi)
                    circ_fixed = tq2qiskit(
                        q_device,
                        q_layer_fixed,
                        remove_ops=self.remove_ops,
                        remove_ops_thres=self.remove_ops_thres,
                    )
                    circ_fixed_list.append(circ_fixed)
                    param.copy_(param + np.pi * 0.5)

        self.transpiled_circs = []
        for circ_fixed in circ_fixed_list:
            circ = QuantumCircuit(q_device.n_wires, q_device.n_wires)
            circ.compose(circ_parameterized, inplace=True)
            circ.compose(circ_fixed, inplace=True)
            v_c_reg_mapping = q_layer_measure.v_c_reg_mapping
            if v_c_reg_mapping is not None:
                for q_reg, c_reg in v_c_reg_mapping["v2c"].items():
                    circ.measure(q_reg, c_reg)
            else:
                circ.measure(
                    list(range(q_device.n_wires)), list(range(q_device.n_wires))
                )

            transpiled_circ = self.transpile(circ)[0]
            self.transpiled_circs.append(transpiled_circ)
        # construct the parameter_binds
        binds_all = []
        if shift_encoder:
            for idx in range(x.size()[1]):
                x[:, idx] += np.pi * 0.5
                for inputs_single in x:
                    binds = {}
                    for k, input_single in enumerate(inputs_single):
                        binds[params[k]] = input_single.item()
                    binds_all.append(binds)

                x[:, idx] -= np.pi
                for inputs_single in x:
                    binds = {}
                    for k, input_single in enumerate(inputs_single):
                        binds[params[k]] = input_single.item()
                    binds_all.append(binds)

                x[:, idx] += np.pi * 0.5
        else:
            for inputs_single in x:
                binds = {}
                for k, input_single in enumerate(inputs_single):
                    binds[params[k]] = input_single.item()
                binds_all.append(binds)

        return self.transpiled_circs, binds_all

    def process_parameterized_and_shift(
        self,
        q_device: tq.QuantumDevice,
        q_layer_parameterized: tq.QuantumModule,
        q_layer_fixed: tq.QuantumModule,
        q_layer_measure: tq.QuantumModule,
        x,
        shift_encoder=False,
        parallel=True,
        shift_this_step=None,
    ):
        """
        separate the conversion, encoder part will be converted to a
        parameterized Qiskit QuantumCircuit. The remaining part will be a
        non-parameterized QuantumCircuit. In this case, only one time of
        compilation is required.

        The returned counts are circuit-major (for each circuit, all
        parameter bindings in order), matching the layout expected by the
        parameter-shift training in tq.node.
        """
        transpiled_circs, binds_all = self.preprocess_parameterized_and_shift(
            q_device,
            q_layer_parameterized,
            q_layer_fixed,
            q_layer_measure,
            x,
            shift_encoder,
            shift_this_step,
        )

        pubs = [(circ, binds) for circ in transpiled_circs for binds in binds_all]
        counts = self._run_pubs_get_counts(pubs, parallel=parallel)

        # kept for API compatibility; V2 primitive jobs are not timed here
        time_spent_list = []

        measured_qiskit = get_expectations_from_counts(counts, n_wires=q_device.n_wires)
        measured_qiskit = torch.tensor(measured_qiskit, dtype=torch.float, device=x.device)

        return measured_qiskit, time_spent_list

    def process_circs_get_joint_expval(self, circs_all, observable, parallel=True):
        """
        This function is used to compute the joint expectation value of a list of observables
        we add diagonalizing gates before sending them to the backend
        """
        observable = observable.upper()
        circs_all_diagonalized = []
        for circ_ in circs_all:
            circ = circ_.copy()
            for k, obs in enumerate(observable):
                if obs == 'X':
                    circ.h(k)
                elif obs == 'Y':
                    circ.z(k)
                    circ.s(k)
                    circ.h(k)
            circ.measure_all()
            circs_all_diagonalized.append(circ)

        expval_all = []

        mask = np.ones(len(observable), dtype=bool)
        mask[np.array([*observable]) == "I"] = False

        counts = self.process_ready_circs_get_counts(circs_all_diagonalized, parallel=parallel)

        # here we need to switch the little and big endian of distribution bitstrings
        distributions = []
        for count in counts:
            distribution = {}
            for k, v in count.items():
                distribution[k[::-1]] = v
            distributions.append(distribution)

        for distri in distributions:
            n_eigen_one = 0
            n_eigen_minus_one = 0
            for bitstring, n_count in distri.items():
                if np.dot(list(map(lambda x: eval(x), [*bitstring])), mask).sum() % 2 == 0:
                    n_eigen_one += n_count
                else:
                    n_eigen_minus_one += n_count

            expval = n_eigen_one / self.n_shots + (-1) * n_eigen_minus_one / self.n_shots
            expval_all.append(expval)

        return expval_all


if __name__ == '__main__':
    circ = QuantumCircuit(3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.rx(0.1, 0)

    qiskit_processor = QiskitProcessor(
        use_real_qc=False
    )

    qiskit_processor.process_ready_circs_get_counts([circ], True)

    qdev = tq.QuantumDevice(n_wires=3, bsz=1)
    qdev.h(0)
    qdev.cx([0, 1])
    qdev.cx([1, 2])
    qdev.rx(0, 0.1)

    from torchquantum.measurement import expval_joint_sampling
    print(expval_joint_sampling(qdev, 'XII', n_shots=8192))

