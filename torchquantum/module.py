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
    def __init__(self) -> None:
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
        """load operation history
            {
                "name": name,  # type: ignore
                "wires": np.array(wires).squeeze().tolist(),
                "params": params.squeeze().detach().cpu().numpy().tolist() if params is not None else None,
                "inverse": inverse,
                "trainable": params.requires_grad if params is not None else False,
            }

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
        """Create a QuantumModule from op_history
        Args:
            op_history (list): A list of op in function dict format
        """
        qmodule = cls()
        qmodule.load_op_history(op_history)
        qmodule.forward = qmodule.forward_Operators_list
        return qmodule

    def forward_Operators_list(self, qdev):
        assert self.Operator_list is not None, "Operator_list should not contain nothing"
        for Oper in self.Operator_list:
            Oper(qdev)

    def set_noise_model_tq(self, noise_model_tq):
        for module in self.modules():
            module.noise_model_tq = noise_model_tq

    def set_qiskit_processor(self, processor):
        for module in self.modules():
            module.qiskit_processor = processor

    def set_wires_per_block(self, wires_per_block):
        self.wires_per_block = wires_per_block

    def static_on(self, is_graph_top=True, wires_per_block=3):
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
        self.static_mode = False
        self.graph = None
        for module in self.children():
            if not isinstance(module, tq.QuantumDevice):
                module.static_off()

    def set_graph_build_finish(self):
        self.graph.is_list_finish = True
        for module in self.graph.module_list:
            if not isinstance(module, tq.QuantumDevice):
                module.set_graph_build_finish()

    def static_forward(self, q_device: tq.QuantumDevice):
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

    def __repr__(self):
        if self.Operator_list is not None:
            return f"QuantumModule with Operator_list {self.Operator_list}"
        else:
            return "QuantumModule"

    def get_unitary(self, q_device: tq.QuantumDevice, x=None):
        original_wires_per_block = self.wires_per_block
        original_static_mode = self.static_mode
        self.static_off()
        self.static_on(wires_per_block=q_device.n_wires)
        self.q_device = q_device
        self.device = q_device.state.device
        self.graph.q_device = q_device
        self.graph.device = q_device.state.device

        self.is_graph_top = False
        # forward to register all modules to the module list, but do not
        # apply the unitary to the state vector
        if x is None:
            self.forward(q_device)
        else:
            self.forward(q_device, x)
        self.is_graph_top = True

        self.graph.build(wires_per_block=q_device.n_wires)
        self.graph.build_static_matrix()
        unitary = self.graph.get_unitary()

        self.static_off()
        if original_static_mode:
            self.static_on(original_wires_per_block)

        return unitary


class QuantumModuleList(nn.ModuleList, QuantumModule, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QuantumModuleDict(nn.ModuleDict, QuantumModule, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



def test():
    pass


if __name__ == "__main__":
    test()
