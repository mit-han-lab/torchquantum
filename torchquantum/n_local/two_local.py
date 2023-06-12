"""The two-local gate circuit."""

from __future__ import annotations
from typing import Union, Optional, List, Callable, Any, Sequence


from torchquantum.devices import QuantumDevice

#Gate in qiskit is a unitary gate with argument "instruction". Parameter is an unknown parameter for gates
# which can possibly be trained. We need to find appropriate replacements
from qiskit.circuit import Gate, Parameter


from .n_local import NLocal
from torchquantum.operators import (
    I,
    PauliZ,
    PauliY,
    PauliZ,
    RX,
    RY,
    RZ,
    H,
    S,
    SDG,
    T,
    TDG,
    RXX,
    RYY,
    RZX,
    RZZ,
    SWAP,
    CNOT,
    CY,
    CZ,
    CRX,
    CRY,
    CRZ,
    CH,
)

class TwoLocal(NLocal):
    r"""The two-local circuit.

    The two-local circuit is a parameterized circuit consisting of alternating rotation layers and
    entanglement layers. The rotation layers are single qubit gates applied on all qubits.
    The entanglement layer uses two-qubit gates to entangle the qubits according to a strategy set
    using ``entanglement``. Both the rotation and entanglement gates can be specified as
    string (e.g. ``'ry'`` or ``'cnot'``), as gate-type (e.g. ``RY`` or ``CNOT``) or
    as QuantumCircuit (e.g. a 1-qubit circuit or 2-qubit circuit).

    A set of default entanglement strategies is provided:

    * ``'full'`` entanglement is each qubit is entangled with all the others.
    * ``'linear'`` entanglement is qubit :math:`i` entangled with qubit :math:`i + 1`,
      for all :math:`i \in \{0, 1, ... , n - 2\}`, where :math:`n` is the total number of qubits.
    * ``'reverse_linear'`` entanglement is qubit :math:`i` entangled with qubit :math:`i + 1`,
      for all :math:`i \in \{n-2, n-3, ... , 1, 0\}`, where :math:`n` is the total number of qubits.
      Note that if ``entanglement_blocks = 'cnot'`` then this option provides the same unitary as
      ``'full'`` with fewer entangling gates.
    * ``'pairwise'`` entanglement is one layer where qubit :math:`i` is entangled with qubit
      :math:`i + 1`, for all even values of :math:`i`, and then a second layer where qubit :math:`i`
      is entangled with qubit :math:`i + 1`, for all odd values of :math:`i`.
    * ``'circular'`` entanglement is linear entanglement but with an additional entanglement of the
      first and last qubit before the linear part.
    * ``'sca'`` (shifted-circular-alternating) entanglement is a generalized and modified version
      of the proposed circuit 14 in `Sim et al. <https://arxiv.org/abs/1905.10876>`__.
      It consists of circular entanglement where the 'long' entanglement connecting the first with
      the last qubit is shifted by one each block.  Furthermore the role of control and target
      qubits are swapped every block (therefore alternating).

    The entanglement can further be specified using an entangler map, which is a list of index
    pairs, such as

    >>> entangler_map = [(0, 1), (1, 2), (2, 0)]

    If different entanglements per block should be used, provide a list of entangler maps.
    See the examples below on how this can be used.

    >>> entanglement = [entangler_map_layer_1, entangler_map_layer_2, ... ]

    Barriers can be inserted in between the different layers for better visualization using the
    ``insert_barriers`` attribute.

    For each parameterized gate a new parameter is generated using a
    :class:`~qiskit.circuit.library.ParameterVector`. The name of these parameters can be chosen
    using the ``parameter_prefix``.

    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        rotation_blocks: Optional[
            Union[str, List[str], type, List[type], QuantumDevice, List[QuantumDevice]]
        ] = None,
        entanglement_blocks: Optional[
            Union[str, List[str], type, List[type], QuantumDevice, List[QuantumDevice]]
        ] = None,
        entanglement: Union[str, List[List[int]], Callable[[int], List[int]]] = "full",
        reps: int = 3,
        skip_unentangled_qubits: bool = False,
        skip_final_rotation_layer: bool = False,
        parameter_prefix: str = "θ",
        insert_barriers: bool = False,
        initial_state: Optional[Any] = None,
        name: str = "TwoLocal",
    ) -> None:
        """Construct a new two-local circuit.

        Args:
            num_qubits: The number of qubits of the two-local circuit.
            rotation_blocks: The gates used in the rotation layer. Can be specified via the name of
                a gate (e.g. ``'ry'``) or the gate type itself (e.g. :class:`.RYGate`).
                If only one gate is provided, the gate same gate is applied to each qubit.
                If a list of gates is provided, all gates are applied to each qubit in the provided
                order.
                See the Examples section for more detail.
            entanglement_blocks: The gates used in the entanglement layer. Can be specified in
                the same format as ``rotation_blocks``.
            entanglement: Specifies the entanglement structure. Can be a string (``'full'``,
                ``'linear'``, ``'reverse_linear'``, ``'circular'`` or ``'sca'``),
                a list of integer-pairs specifying the indices
                of qubits entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                Default to ``'full'`` entanglement.
                Note that if ``entanglement_blocks = 'cx'``, then ``'full'`` entanglement provides the
                same unitary as ``'reverse_linear'`` but the latter option has fewer entangling gates.
                See the Examples section for more detail.
            reps: Specifies how often a block consisting of a rotation layer and entanglement
                layer is repeated.
            skip_unentangled_qubits: If ``True``, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If ``False``, the single qubit gates are applied
                to each qubit in the ansatz. Defaults to ``False``.
            skip_final_rotation_layer: If ``False``, a rotation layer is added at the end of the
                ansatz. If ``True``, no rotation layer is added.
            parameter_prefix: The parameterized gates require a parameter to be defined, for which
                we use instances of :class:`~qiskit.circuit.Parameter`. The name of each parameter will
                be this specified prefix plus its index.
            insert_barriers: If ``True``, barriers are inserted in between each layer. If ``False``,
                no barriers are inserted. Defaults to ``False``.
            initial_state: A :class:`.QuantumCircuit` object to prepend to the circuit.

        """
        super().__init__(
            num_qubits=num_qubits,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement=entanglement,
            reps=reps,
            skip_final_rotation_layer=skip_final_rotation_layer,
            skip_unentangled_qubits=skip_unentangled_qubits,
            insert_barriers=insert_barriers,
            initial_state=initial_state,
            parameter_prefix=parameter_prefix,
            name=name,
        )

    def _convert_to_block(self, layer: Union[str, type, Gate, QuantumDevice]) -> QuantumDevice:
        """For a layer provided as str (e.g. ``'ry'``) or type (e.g. :class:`.RYGate`) this function
         returns the
         according layer type along with the number of parameters (e.g. ``(RYGate, 1)``).

        Args:
            layer: The qubit layer.

        Returns:
            The specified layer with the required number of parameters.

        Raises:
            TypeError: The type of ``layer`` is invalid.
            ValueError: The type of ``layer`` is str but the name is unknown.
            ValueError: The type of ``layer`` is type but the layer type is unknown.

        Note:
            Outlook: If layers knew their number of parameters as static property, we could also
            allow custom layer types.
        """
        if isinstance(layer, QuantumDevice):
            return layer

        # check the list of valid layers
        # this could be a lot easier if the standard layers would have ``name`` and ``num_params``
        # as static types, which might be something they should have anyway
        theta = Parameter("θ")
        valid_layers = {
            "ch": CH(),
            "cx": CNOT(),
            "cnot": CNOT(),
            "cy": CY(),
            "cz": CZ(),
            "crx": CRX(theta),
            "cry": CRY(theta),
            "crz": CRZ(theta),
            "h": H(),
            "i": I(),
            "id": I(),
            "iden": I(),
            "rx": RX(theta),
            "rxx": RXX(theta),
            "ry": RY(theta),
            "ryy": RYY(theta),
            "rz": RZ(theta),
            "rzx": RZX(theta),
            "rzz": RZZ(theta),
            "s": S(),
            "sdg": SDG(),
            "swap": SWAP(),
            "x": X(),
            "y": Y(),
            "z": Z(),
            "t": T(),
            "tdg": TDG(),
        }

        # try to exchange `layer` from a string to a gate instance
        if isinstance(layer, str):
            try:
                layer = valid_layers[layer]
            except KeyError as ex:
                raise ValueError(f"Unknown layer name `{layer}`.") from ex

        # try to exchange `layer` from a type to a gate instance
        if isinstance(layer, type):
            # iterate over the layer types and look for the specified layer
            instance = None
            for gate in valid_layers.values():
                if isinstance(gate, layer):
                    instance = gate
            if instance is None:
                raise ValueError(f"Unknown layer type`{layer}`.")
            layer = instance

        if isinstance(layer):
            circuit = QuantumDevice(layer.num_qubits)
            circuit.append(layer, list(range(layer.num_qubits)))
            return circuit

        raise TypeError(
            f"Invalid input type {type(layer)}. " + "`layer` must be a type, str or QuantumDevice."
        )

    def get_entangler_map(
        self, rep_num: int, block_num: int, num_block_qubits: int
    ) -> Sequence[Sequence[int]]:
        """Overloading to handle the special case of 1 qubit where the entanglement are ignored."""
        if self.num_qubits <= 1:
            return []
        return super().get_entangler_map(rep_num, block_num, num_block_qubits)
