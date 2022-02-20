# FIXED parameterization

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Qiskit Aer qasm simulator backend.
"""

import copy
import datetime
import logging
import time
import uuid
import warnings
from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit, ParameterExpression, Delay
from qiskit.compiler import assemble
from qiskit.providers import BackendV1 as Backend
from qiskit.providers.models import BackendStatus
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.qobj import QasmQobj, PulseQobj
from qiskit.result import Result
from qiskit.utils import deprecate_arguments
from ..aererror import AerError
from ..jobs import AerJob, AerJobSet, split_qobj
from ..noise.noise_model import NoiseModel, QuantumErrorLocation
from ..noise.errors.quantum_error import QuantumChannelInstruction
from .aer_compiler import compile_circuit
from .backend_utils import format_save_type, circuit_optypes

# Logger
logger = logging.getLogger(__name__)


class AerBackend(Backend, ABC):
    """Qiskit Aer Backend class."""

    def __init__(self,
                 configuration,
                 properties=None,
                 defaults=None,
                 backend_options=None,
                 provider=None):
        """Aer class for backends.

        This method should initialize the module and its configuration, and
        raise an exception if a component of the module is
        not available.

        Args:
            configuration (BackendConfiguration): backend configuration.
            properties (BackendProperties or None): Optional, backend properties.
            defaults (PulseDefaults or None): Optional, backend pulse defaults.
            provider (Provider): Optional, provider responsible for this backend.
            backend_options (dict or None): Optional set custom backend options.

        Raises:
            AerError: if there is no name in the configuration
        """
        # Init configuration and provider in Backend
        configuration.simulator = True
        configuration.local = True
        super().__init__(configuration, provider=provider)

        # Initialize backend properties and pulse defaults.
        self._properties = properties
        self._defaults = defaults

        # Custom option values for config, properties, and defaults
        self._options_configuration = {}
        self._options_defaults = {}
        self._options_properties = {}

        # Set options from backend_options dictionary
        if backend_options is not None:
            self.set_options(**backend_options)

    def _convert_circuit_binds(self, circuit, binds):
        parameterizations = []
        for index, inst_tuple in enumerate(circuit.data):
            if inst_tuple[0].is_parameterized():
                for bind_pos, param in enumerate(inst_tuple[0].params):
                    if param in binds:
                        parameterizations.append([[index, bind_pos], binds[param]])
                    elif isinstance(param, ParameterExpression):
                        # If parameter expression has no unbound parameters
                        # it's already bound and should be skipped
                        if not param.parameters:
                            continue
                        local_binds = {k: v for k, v in binds.items() if k in param.parameters}
                        bind_list = [dict(zip(local_binds, t)) for t in zip(*local_binds.values())]
                        bound_values = [float(param.bind(x)) for x in bind_list]
                        parameterizations.append([[index, bind_pos], bound_values])
        return parameterizations

    def _convert_binds(self, circuits, parameter_binds):
        if isinstance(circuits, QuantumCircuit):
            if len(parameter_binds) > 1:
                raise AerError("More than 1 parameter table provided for a single circuit")

            return [self._convert_circuit_binds(circuits, parameter_binds[0])]
        elif len(parameter_binds) != len(circuits):
            raise AerError(
                "Number of input circuits does not match number of input "
                "parameter bind dictionaries"
            )
        parameterizations = [
            self._convert_circuit_binds(
                circuit, parameter_binds[idx]) for idx, circuit in enumerate(circuits)
        ]
        return parameterizations

    # pylint: disable=arguments-differ
    @deprecate_arguments({'qobj': 'circuits'})
    def run(self,
            circuits,
            validate=False,
            parameter_binds=None,
            **run_options):
        """Run a qobj on the backend.

        Args:
            circuits (QuantumCircuit or list): The QuantumCircuit (or list
                of QuantumCircuit objects) to run
            validate (bool): validate the Qobj before running (default: False).
            parameter_binds (list): A list of parameter binding dictionaries.
                                    See additional information (default: None).
            run_options (kwargs): additional run time backend options.

        Returns:
            AerJob: The simulation job.

        Raises:
            AerError: If ``parameter_binds`` is specified with a qobj input or has a
                length mismatch with the number of circuits.

        Additional Information:
            * Each parameter binding dictionary is of the form::

                {
                    param_a: [val_1, val_2],
                    param_b: [val_3, val_1],
                }

              for all parameters in that circuit. The length of the value
              list must be the same for all parameters, and the number of
              parameter dictionaries in the list must match the length of
              ``circuits`` (if ``circuits`` is a single ``QuantumCircuit``
              object it should a list of length 1).
            * kwarg options specified in ``run_options`` will temporarily override
              any set options of the same name for the current run.

        Raises:
            ValueError: if run is not implemented
        """
        if isinstance(circuits, (QasmQobj, PulseQobj)):
            warnings.warn(
                'Using a qobj for run() is deprecated as of qiskit-aer 0.9.0'
                ' and will be removed no sooner than 3 months from that release'
                ' date. Transpiled circuits should now be passed directly using'
                ' `backend.run(circuits, **run_options).',
                DeprecationWarning, stacklevel=2)
            if parameter_binds:
                raise AerError("Parameter binds can't be used with an input qobj")
            # A work around to support both qobj options and run options until
            # qobj is deprecated is to copy all the set qobj.config fields into
            # run_options that don't override existing fields. This means set
            # run_options fields will take precidence over the value for those
            # fields that are set via assemble.
            if not run_options:
                run_options = circuits.config.__dict__
            else:
                run_options = copy.copy(run_options)
                for key, value in circuits.config.__dict__.items():
                    if key not in run_options and value is not None:
                        run_options[key] = value
            qobj = self._assemble(circuits, **run_options)
        else:
            qobj = self._assemble(circuits, parameter_binds=parameter_binds, **run_options)

        # Optional validation
        if validate:
            self._validate(qobj)

        # Get executor from qobj config and delete attribute so qobj can still be serialized
        executor = getattr(qobj.config, 'executor', None)
        if hasattr(qobj.config, 'executor'):
            delattr(qobj.config, 'executor')

        # Optionally split the job
        experiments = split_qobj(
            qobj, max_size=getattr(qobj.config, 'max_job_size', None),
            max_shot_size=getattr(qobj.config, 'max_shot_size', None))

        # Temporarily remove any executor from options so that job submission
        # can work with Dask client executors which can't be pickled
        opts_executor = getattr(self._options, 'executor', None)
        if hasattr(self._options, 'executor'):
            self._options.executor = None

        # Submit job
        job_id = str(uuid.uuid4())
        if isinstance(experiments, list):
            aer_job = AerJobSet(self, job_id, self._run, experiments, executor)
        else:
            aer_job = AerJob(self, job_id, self._run, experiments, executor)
        aer_job.submit()

        # Restore removed executor after submission
        if hasattr(self._options, 'executor'):
            self._options.executor = opts_executor

        return aer_job

    def configuration(self):
        """Return the simulator backend configuration.

        Returns:
            BackendConfiguration: the configuration for the backend.
        """
        config = copy.copy(self._configuration)
        for key, val in self._options_configuration.items():
            setattr(config, key, val)
        # If config has custom instructions add them to
        # basis gates to include them for the terra transpiler
        if hasattr(config, 'custom_instructions'):
            config.basis_gates = config.basis_gates + config.custom_instructions
        return config

    def properties(self):
        """Return the simulator backend properties if set.

        Returns:
            BackendProperties: The backend properties or ``None`` if the
                               backend does not have properties set.
        """
        properties = copy.copy(self._properties)
        for key, val in self._options_properties.items():
            setattr(properties, key, val)
        return properties

    def defaults(self):
        """Return the simulator backend pulse defaults.

        Returns:
            PulseDefaults: The backend pulse defaults or ``None`` if the
                           backend does not support pulse.
        """
        defaults = copy.copy(self._defaults)
        for key, val in self._options_defaults.items():
            setattr(defaults, key, val)
        return defaults

    @classmethod
    def _default_options(cls):
        pass

    def clear_options(self):
        """Reset the simulator options to default values."""
        self._options = self._default_options()
        self._options_configuration = {}
        self._options_properties = {}
        self._options_defaults = {}

    def status(self):
        """Return backend status.

        Returns:
            BackendStatus: the status of the backend.
        """
        return BackendStatus(
            backend_name=self.name(),
            backend_version=self.configuration().backend_version,
            operational=True,
            pending_jobs=0,
            status_msg='')

    def _run(self, qobj, job_id='', format_result=True):
        """Run a job"""
        # Start timer
        start = time.time()

        # Run simulation
        output = self._execute(qobj)

        # Validate output
        if not isinstance(output, dict):
            logger.error("%s: simulation failed.", self.name())
            if output:
                logger.error('Output: %s', output)
            raise AerError(
                "simulation terminated without returning valid output.")

        # Format results
        output["job_id"] = job_id
        output["date"] = datetime.datetime.now().isoformat()
        output["backend_name"] = self.name()
        output["backend_version"] = self.configuration().backend_version

        # Add execution time
        output["time_taken"] = time.time() - start

        # Display warning if simulation failed
        if not output.get("success", False):
            msg = "Simulation failed"
            if "status" in output:
                msg += f" and returned the following error message:\n{output['status']}"
            logger.warning(msg)
        if format_result:
            return self._format_results(output)
        return output

    @staticmethod
    def _format_results(output):
        """Format C++ simulator output for constructing Result"""
        for result in output["results"]:
            data = result.get("data", {})
            metadata = result.get("metadata", {})
            save_types = metadata.get("result_types", {})
            save_subtypes = metadata.get("result_subtypes", {})
            for key, val in data.items():
                if key in save_types:
                    data[key] = format_save_type(val, save_types[key], save_subtypes[key])
        return Result.from_dict(output)

    def _assemble(self, circuits, parameter_binds=None, **run_options):
        """Assemble one or more Qobj for running on the simulator"""
        if isinstance(circuits, (QasmQobj, PulseQobj)):
            qobj = circuits
        else:
            # Generate optypes for circuit
            # Generate opsets of instructions
            if isinstance(circuits, (QuantumCircuit, Schedule, ScheduleBlock)):
                circuits = [circuits]
            optypes = [circuit_optypes(circ) for circ in circuits]

            # Compile Qasm3 instructions
            circuits, optypes = compile_circuit(
                circuits,
                basis_gates=self.configuration().basis_gates,
                optypes=optypes)

            # run option noise model
            circuits, optypes, run_options = self._assemble_noise_model(
                circuits, optypes, **run_options)

            if parameter_binds:
                # Handle parameter binding
                # parameterizations = self._convert_binds(circuits, parameter_binds)
                # assemble_binds = []
                # assemble_binds.append({param: 1 for bind in parameter_binds for param in bind})

                # qobj = assemble(
                #     circuits,
                #     backend=self,
                #     parameter_binds=assemble_binds,
                #     parameterizations=parameterizations)
                qobj = assemble(circuits, self, parameter_binds=parameter_binds)
            else:
                qobj = assemble(circuits, backend=self)

            # Add optypes to qobj
            # We convert to strings to avoid pybinding of types
            qobj.config.optypes = [
                set(i.__name__ for i in optype) if optype else set()
                for optype in optypes]

        # Add options
        for key, val in self.options.__dict__.items():
            if val is not None:
                setattr(qobj.config, key, val)

        # Override with run-time options
        for key, val in run_options.items():
            setattr(qobj.config, key, val)

        return qobj

    def _assemble_noise_model(self, circuits, optypes, **run_options):
        """Move quantum error instructions from circuits to noise model"""
        if isinstance(circuits, (QuantumCircuit, Schedule, ScheduleBlock)):
            run_circuits = [circuits]
        else:
            # Make a shallow copy so we can modify list elements if required
            run_circuits = copy.copy(circuits)

        # Flag for if we need to make a deep copy of the noise model
        # This avoids unnecessarily copying the noise model for circuits
        # that do not contain a quantum error
        updated_noise = False
        noise_model = run_options.get(
            'noise_model', getattr(self.options, 'noise_model', None))

        # Add custom pass noise only to QuantumCircuit objects that contain delay
        # instructions since this is the only instruction handled by the noise pass
        # at present
        if noise_model and all(isinstance(circ, QuantumCircuit) for circ in run_circuits):
            npm = noise_model._pass_manager()
            if npm is not None:
                # Get indicies of circuits that need noise transpiling
                transpile_idxs = [
                    idx for idx, optype in enumerate(optypes) if Delay in optype
                ]

                # Transpile only the required circuits
                transpiled_circuits = npm.run([run_circuits[i] for i in transpile_idxs])
                if isinstance(transpiled_circuits, QuantumCircuit):
                    transpiled_circuits = [transpiled_circuits]

                # Update the circuits with transpiled ones
                for idx, circ in zip(transpile_idxs, transpiled_circuits):
                    run_circuits[idx] = circ
                    optypes[idx] = circuit_optypes(circ)

        # Check if circuits contain quantum error instructions
        for idx, circ in enumerate(run_circuits):
            if QuantumChannelInstruction in optypes[idx] and not isinstance(
                    circ, (Schedule, ScheduleBlock)):
                updated_circ = False
                new_data = []
                for inst, qargs, cargs in circ.data:
                    if isinstance(inst, QuantumChannelInstruction):
                        updated_circ = True
                        if not updated_noise:
                            # Deep copy noise model on first update
                            if noise_model is None:
                                noise_model = NoiseModel()
                            else:
                                noise_model = copy.deepcopy(noise_model)
                            updated_noise = True
                        # Extract error and replace with place holder
                        qerror = inst._quantum_error
                        qerror_loc = QuantumErrorLocation(qerror)
                        new_data.append((qerror_loc, qargs, cargs))
                        optypes[idx].add(QuantumErrorLocation)
                        # Add error to noise model
                        if qerror.id not in noise_model._default_quantum_errors:
                            noise_model.add_all_qubit_quantum_error(qerror, qerror.id)
                    else:
                        new_data.append((inst, qargs, cargs))
                if updated_circ:
                    new_circ = circ.copy()
                    new_circ.data = new_data
                    run_circuits[idx] = new_circ
                    optypes[idx].discard(QuantumChannelInstruction)

        # If we modified the existing noise model, add it to the run options
        if updated_noise:
            run_options['noise_model'] = noise_model

        # Return the possibly updated circuits and noise model
        return run_circuits, optypes, run_options

    def _get_executor(self, **run_options):
        """Get the executor"""
        if 'executor' in run_options:
            return run_options['executor']
        else:
            return getattr(self._options, 'executor', None)

    @abstractmethod
    def _execute(self, qobj):
        """Execute a qobj on the backend.

        Args:
            qobj (QasmQobj or PulseQobj): simulator input.

        Returns:
            dict: return a dictionary of results.
        """
        pass

    def _validate(self, qobj):
        """Validate the qobj for the backend"""
        pass

    def set_option(self, key, value):
        """Special handling for setting backend options.

        This method should be extended by sub classes to
        update special option values.

        Args:
            key (str): key to update
            value (any): value to update.

        Raises:
            AerError: if key is 'method' and val isn't in available methods.
        """
        # Add all other options to the options dict
        # TODO: in the future this could be replaced with an options class
        #       for the simulators like configuration/properties to show all
        #       available options
        if hasattr(self._configuration, key):
            self._set_configuration_option(key, value)
        elif hasattr(self._properties, key):
            self._set_properties_option(key, value)
        elif hasattr(self._defaults, key):
            self._set_defaults_option(key, value)
        else:
            if not hasattr(self._options, key):
                raise AerError("Invalid option %s" % key)
            if value is not None:
                # Only add an option if its value is not None
                setattr(self._options, key, value)
            else:
                # If setting an existing option to None reset it to default
                # this is for backwards compatibility when setting it to None would
                # remove it from the options dict
                setattr(self._options, key, getattr(self._default_options(), key))

    def set_options(self, **fields):
        """Set the simulator options"""
        for key, value in fields.items():
            self.set_option(key, value)

    def _set_configuration_option(self, key, value):
        """Special handling for setting backend configuration options."""
        if value is not None:
            self._options_configuration[key] = value
        elif key in self._options_configuration:
            self._options_configuration.pop(key)

    def _set_properties_option(self, key, value):
        """Special handling for setting backend properties options."""
        if value is not None:
            self._options_properties[key] = value
        elif key in self._options_properties:
            self._options_properties.pop(key)

    def _set_defaults_option(self, key, value):
        """Special handling for setting backend defaults options."""
        if value is not None:
            self._options_defaults[key] = value
        elif key in self._options_defaults:
            self._options_defaults.pop(key)

    def __repr__(self):
        """String representation of an AerBackend."""
        name = self.__class__.__name__
        display = f"'{self.name()}'"
        return f'{name}({display})'
