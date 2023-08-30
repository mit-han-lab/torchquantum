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

import torch.nn as nn
import torchquantum as tq

from abc import ABCMeta
from typing import Iterable

__all__ = [
    "QuantumModule",
    "QuantumModuleList",
    "QuantumModuleDict",
]


class QuantumModule(nn.Module):
    """Module for quantum computations.
    
    Attributes:
        static_mode (bool): Indicates whether the module is in static mode.
        graph (tq.QuantumGraph): The quantum graph representation.
        parent_graph (tq.QuantumGraph): The parent quantum graph.
        is_graph_top (bool): Indicates whether the module is the top level of the graph.
        unitary (torch.Tensor): The unitary matrix.
        wires (List[int]): The wires.
        n_wires (int): The number of wires.
        q_device (tq.QuantumDevice): The quantum device.
        device (torch.device): The device (CPU or GPU).
        wires_per_block (int): The number of wires per block for static tensor network simulation.
        qiskit_processor: The Qiskit processor.
        noise_model_tq: The noise model.
        Operator_list (tq.QuantumModuleList): The list of quantum operators.
    
    Methods:
        __init__(self):
            Initialize the QuantumModule.
        load_op_history(self, op_history):
            Load the operation history.
        from_op_history(cls, op_history):
            Create a QuantumModule from the operation history.
    """
    
    def __init__(self) -> None:
        """Initialize the QuantumModule.
        
        Returns:
            None.
            
        Examples:
            >>> qmodule = QuantumModule()
        """
        
        super().__init__()
        self.static_mode = False
        self.graph = None
        self.parent_graph = None
        self.is_graph_top = False
        self.unitary = None
        self.wires = None
        self.n_wires = None
        self.q_device = None
        # this is for gpu or cpu, not q device
        self.device = None
        # for the static tensor network simulation optimizations
        self.wires_per_block = None
        self.qiskit_processor = None
        self.noise_model_tq = None
        self.Operator_list = None
    
    def load_op_history(self, op_history):
        """Load the operation history.
        
        Args:
            op_history (list of dict): A list of operations in function dict format.

        Returns:
            None.
            
        Examples:
            >>> op_history = [
            ...     {
            ...         "name": name,   # type: ignore
            ...         "wires": np.array(wires).squeeze().tolist(),
            ...         "params": params.squeeze().detach().cpu().numpy().tolist() if params is not None else None,
            ...         "inverse": inverse,
            ...         "trainable": params.requires_grad if params is not None else False,
            ...     }
            ... ]
            >>> qmodule.load_op_history(op_history)
        """

        Operator_list = []
        for op in op_history:
            Oper = tq.op_name_dict[op["name"]]
            trainable = op.get("trainable", False)
            has_params = True if ((op.get("params", None) is not None) or trainable) else False
            init_params = op.get("params", None)
            n_wires = len(op["wires"]) if isinstance(op["wires"], Iterable) else 1
            wires = op['wires']
            inverse = op.get("inverse", False)
            Operator_list.append(Oper(
                has_params=has_params,
                trainable=trainable,
                init_params=init_params,
                n_wires=n_wires,
                wires=wires,
                inverse=inverse,
            ))

        self.Operator_list = tq.QuantumModuleList(Operator_list)
    

    @classmethod
    def from_op_history(cls, op_history):
        """Create a QuantumModule from the operation history.

        Args:
            op_history (list): A list of operations in function dict format.

        Returns:
            QuantumModule: The created QuantumModule.
        
        Examples:
            >>> op_history = [
            ...     {
            ...         "name": name,   # type: ignore
            ...         "wires": np.array(wires).squeeze().tolist(),
            ...         "params": params.squeeze().detach().cpu().numpy().tolist() if params is not None else None,
            ...         "inverse": inverse,
            ...         "trainable": params.requires_grad if params is not None else False,
            ...     }
            ... ]
            >>> qmodule = QuantumModule.from_op_history(op_history)
        """
        
        qmodule = cls()
        qmodule.load_op_history(op_history)
        qmodule.forward = qmodule.forward_Operators_list
        return qmodule

    def forward_Operators_list(self, qdev):
        """Forward the list of quantum operators.

        Args:
            qdev (tq.QuantumDevice): The quantum device.
        
        Returns:
            None.

        Examples:
            >>> qdev = tq.QuantumDevice(n_wires=2)
            >>> qmodule.forward_Operators_list(qdev)
        """
        
        assert self.Operator_list is not None, "Operator_list should not contain nothing"
        for Oper in self.Operator_list:
            Oper(qdev)

    def set_noise_model_tq(self, noise_model_tq):
        """Set the noise model for the QuantumModule and its sub-modules.

        Args:
            noise_model_tq: The noise model.
        
        Returns:
            None.
            
        Examples:
            >>> noise_model_tq = ...
            >>> qmodule.set_noise_model_tq(noise_model_tq)
        """
        
        for module in self.modules():
            module.noise_model_tq = noise_model_tq

    def set_qiskit_processor(self, processor):
        """Set the Qiskit processor for the QuantumModule and its sub-modules.

        Args:
            processor: The Qiskit processor.

        Returns:
            None.

        Examples:
            >>> processor = ...
            >>> qmodule.set_qiskit_processor(processor)
        """
        
        for module in self.modules():
            module.qiskit_processor = processor

    def set_wires_per_block(self, wires_per_block):
        """Set the number of wires per block for static tensor network simulation.

        Args:
            wires_per_block (int): The number of wires per block.

        Returns:
            None.

        Examples:
            >>> qmodule.set_wires_per_block(3)
        """
        
        self.wires_per_block = wires_per_block

    def static_on(self, is_graph_top=True, wires_per_block=3):
        """Enable static mode for the QuantumModule.

        Args:
            is_graph_top (bool): Indicates whether the module is the top level of the graph.
                Defaults to True.
            wires_per_block (int): The number of wires per block.
                Defaults to 3.

        Returns:
            None.

        Examples:
            >>> qmodule.static_on(wires_per_block=5)
        """
        
        self.wires_per_block = wires_per_block
        # register graph of itself and parent
        self.static_mode = True
        self.is_graph_top = is_graph_top
        if self.graph is None:
            self.graph = tq.QuantumGraph()

        for module in self.children():
            if isinstance(module, nn.ModuleList) or isinstance(module, nn.ModuleDict):
                # if QuantumModuleList or QuantumModuleDict, its graph will
                # be the same as the parent graph because ModuleList and
                # ModuleDict do not call the forward function
                module.graph = self.graph
            module.parent_graph = self.graph
            if not isinstance(module, tq.QuantumDevice):
                module.static_on(is_graph_top=False)

    def static_off(self):
        """Disable static mode for the QuantumModule.

        Returns:
            None.

        Examples:
            >>> qmodule.static_off()
        """
        
        self.static_mode = False
        self.graph = None
        for module in self.children():
            if not isinstance(module, tq.QuantumDevice):
                module.static_off()

    def set_graph_build_finish(self):
        """Set the graph build finish flag for the QuantumModule and its sub-modules.
        
        Returns:
            None.        

        Examples:
            >>> qmodule.set_graph_build_finish()
        """
        
        self.graph.is_list_finish = True
        for module in self.graph.module_list:
            if not isinstance(module, tq.QuantumDevice):
                module.set_graph_build_finish()

    def static_forward(self, q_device: tq.QuantumDevice):
        """Static forward pass of the QuantumModule with the given quantum device.

        Args:
            q_device (tq.QuantumDevice): The quantum device.
        
        Returns:
            None.
            
        Example:
            >>> q_device = tq.QuantumDevice(n_wires=2)
            >>> qmodule.static_forward(q_device)
        """
        
        self.q_device = q_device
        self.device = q_device.states.device
        self.graph.q_device = q_device
        self.graph.device = q_device.states.device
        # self.unitary, self.wires, self.n_wires = \
        self.graph.forward(wires_per_block=self.wires_per_block)
        # tqf.qubitunitary(
        #     q_device=self.q_device,
        #     wires=self.wires,
        #     params=self.unitary
        # )

    # def __repr__(self):
    #     if self.Operator_list is not None:
    #         return f"QuantumModule with Operator_list {self.Operator_list}"
    #     else:
    #         return "QuantumModule"

    def get_unitary(self, x=None):
        """Compute the unitary matrix for the QuantumModule with the given quantum device and input.

        Args:
            q_device (tq.QuantumDevice): The quantum device.
            x (Optional): The input.
                Defaults to None.

        Returns:
            torch.Tensor: The unitary matrix.

        Example:
            >>> q_device = tq.QuantumDevice(n_wires=2)
            >>> unitary = qmodule.get_unitary(q_device)
        """
        assert self.n_wires is not None, "n_wires should not be None, specify it in the Quantum Module \
            before calling get_unitary()" 
        
        qdev = tq.QuantumDevice(n_wires=self.n_wires)
        qdev.reset_identity_states()
    
        if x is None:
            self.forward(qdev)
        else:
            self.forward(qdev, x)
        
        unitary = qdev.get_states_1d().T

        return unitary


class QuantumModuleList(nn.ModuleList, QuantumModule, metaclass=ABCMeta):
    """A list-based container for QuantumModules."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QuantumModuleDict(nn.ModuleDict, QuantumModule, metaclass=ABCMeta):
    """A dictionary-based container for QuantumModules."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



def test():
    """Test function.
    
    Returns:
        None.
    """
    
    pass


if __name__ == "__main__":
    test()
