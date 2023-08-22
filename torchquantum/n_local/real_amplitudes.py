
from typing import Union, Optional, List, Tuple, Callable, Any
import numpy as np

from torchquantum.operators import RY, CNOT     
from .two_local import TwoLocal

class RealAmplitudes(TwoLocal):
    r"""The real-amplitudes 2-local circuit.

    The ``RealAmplitudes`` circuit is a heuristic trial wave function used as Ansatz in chemistry
    applications or classification circuits in machine learning. The circuit consists of
    of alternating layers of :math:`Y` rotations and :math:`CNOT` entanglements. The entanglement
    pattern can be user-defined or selected from a predefined set.
    It is called ``RealAmplitudes`` since the prepared quantum states will only have
    real amplitudes, the complex part is always 0.

    The entanglement can be set using the ``entanglement`` keyword as string or a list of
    index-pairs.Additional options that can be set include the
    number of repetitions, skipping rotation gates on qubits that are not entangled, leaving out
    the final rotation layer and inserting barriers in between the rotation and entanglement
    layers.

    If some qubits are not entangled with other qubits it makes sense to not apply rotation gates
    on these qubits, since a sequence of :math:`Y` rotations can be reduced to a single :math:`Y`
    rotation with summed rotation angles.
    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "reverse_linear",
        reps: int = 3,
        skip_unentangled_qubits: bool = False,
        skip_final_rotation_layer: bool = False,
        parameter_prefix: str = "θ",
        insert_barriers: bool = False,
        initial_state: Optional[Any] = None,
        name: str = "RealAmplitudes",
    ) -> None:
        """Create a new RealAmplitudes 2-local circuit.

        Args:
            num_qubits: The number of qubits of the RealAmplitudes circuit.
            reps: Specifies how often the structure of a rotation layer followed by an entanglement
                layer is repeated.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                'reverse_linear, 'circular' or 'sca'), a list of integer-pairs specifying the indices
                of qubits entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'reverse_linear' entanglement.
                Note that 'reverse_linear' entanglement provides the same unitary as 'full'
                with fewer entangling gates.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_final_rotation_layer: If False, a rotation layer is added at the end of the
                ansatz. If True, no rotation layer is added.
            parameter_prefix: The parameterized gates require a parameter to be defined
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted.

        """
        super().__init__(
            num_qubits=num_qubits,
            reps=reps,
            rotation_blocks=RY,
            entanglement_blocks=CNOT,
            entanglement=entanglement,
            initial_state=initial_state,
            skip_unentangled_qubits=skip_unentangled_qubits,
            skip_final_rotation_layer=skip_final_rotation_layer,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Return the parameter bounds.

        Returns:
            The parameter bounds.
        """
        return self.num_parameters * [(-np.pi, np.pi)]
