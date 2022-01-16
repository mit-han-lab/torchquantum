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
import json
import logging
import datetime
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from numpy import ndarray

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.providers import BackendV1 as Backend
from qiskit.providers.models import BackendStatus
from qiskit.result import Result
from qiskit.utils import deprecate_arguments
from qiskit.qobj import QasmQobj, PulseQobj
from qiskit.compiler import assemble

from ..jobs import AerJob, AerJobSet, split_qobj
from ..aererror import AerError


# Logger
logger = logging.getLogger(__name__)


class AerJSONEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy arrays and complex numbers.

    This functions as the standard JSON Encoder but adds support
    for encoding:
        complex numbers z as lists [z.real, z.imag]
        ndarrays as nested lists.
    """

    # pylint: disable=method-hidden,arguments-differ
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


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
        experiments = split_qobj(qobj, max_size=getattr(qobj.config, 'max_job_size', None))

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

    def _run(self, qobj, job_id=''):
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

        return Result.from_dict(output)

    def _assemble(self, circuits, parameter_binds=None, **run_options):
        """Assemble one or more Qobj for running on the simulator"""
        # This conditional check can be removed when we remove passing
        # qobj to run
        if isinstance(circuits, (QasmQobj, PulseQobj)):
            qobj = circuits
        elif parameter_binds:
            # Handle parameter binding
            # parameterizations = self._convert_binds(circuits, parameter_binds)
            # assemble_binds = []
            # assemble_binds.append({param: 1 for bind in parameter_binds for param in bind})

            qobj = assemble(circuits, self, parameter_binds=parameter_binds)
        else:
            qobj = assemble(circuits, self)

        # Add options
        for key, val in self.options.__dict__.items():
            if val is not None:
                setattr(qobj.config, key, val)

        # Override with run-time options
        for key, val in run_options.items():
            setattr(qobj.config, key, val)

        return qobj

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
