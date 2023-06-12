
#The efficient SU2 2-local circuit

from typing import Union, Optional, List, Tuple, Callable, Any
from numpy import pi

from torchquantum.devices import QuantumDevice
from torchquantum.operators import RY, RZ, CNOT
from .two_local import TwoLocal

class EfficientSU2(TwoLocal):


    r"""The hardware efficient SU(2) 2-local circuit.

    The ``EfficientSU2`` circuit consists of layers of single qubit operations spanned by SU(2)
    and :math:`CX` entanglements. This is a heuristic pattern that can be used to prepare trial wave
    functions for variational quantum algorithms or classification circuit for machine learning.

    SU(2) stands for special unitary group of degree 2, its elements are :math:`2 \times 2`
    unitary matrices with determinant 1, such as the Pauli rotation gates.

    On 3 qubits and using the Pauli :math:`Y` and :math:`Z` su2_gates as single qubit gates, the
    hardware efficient SU(2) circuit is represented by:
    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        su2_gates: Optional[
            Union[
                str,
                type,
                QuantumDevice,
                List[Union[str, type,QuantumDevice]],
            ]
        ] = None,
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "reverse_linear",
        reps: int = 3,
        skip_unentangled_qubits: bool = False,
        skip_final_rotation_layer: bool = False,
        parameter_prefix: str = "θ",
        insert_barriers: bool = False,
        initial_state: Optional[Any] = None,
        name: str = "EfficientSU2",
    ) -> None:
        """Create a new EfficientSU2 2-local circuit.

        Args:
            num_qubits: The number of qubits of the EfficientSU2 circuit.
            reps: Specifies how often the structure of a rotation layer followed by an entanglement
                layer is repeated.
            su2_gates: The SU(2) single qubit gates to apply in single qubit gate layers.
                If only one gate is provided, the same gate is applied to each qubit.
                If a list of gates is provided, all gates are applied to each qubit in the provided
                order.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                , 'reverse_linear', 'circular' or 'sca'), a list of integer-pairs specifying the indices
                of qubits entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to 'reverse_linear' entanglement.
                Note that 'reverse_linear' entanglement provides the same unitary as 'full'
                with fewer entangling gates.
            initial_state: A `QuantumDevice` object to prepend to the circuit.
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_final_rotation_layer: If False, a rotation layer is added at the end of the
                ansatz. If True, no rotation layer is added.
            parameter_prefix: The parameterized gates require a parameter to be defined
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted.

        """
        if su2_gates is None:
            su2_gates = [RY, RZ]
        super().__init__(
            num_qubits=num_qubits,
            rotation_blocks=su2_gates,
            entanglement_blocks=CNOT    ,
            entanglement=entanglement,
            reps=reps,
            skip_unentangled_qubits=skip_unentangled_qubits,
            skip_final_rotation_layer=skip_final_rotation_layer,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            initial_state=initial_state,
            name=name,
        )

    @property
    def parameter_bounds(self) -> List[Tuple[float, float]]:
        """Return the parameter bounds.

        Returns:
            The parameter bounds.
        """
        return self.num_parameters * [(-pi, pi)]
