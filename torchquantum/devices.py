# import copy
# import time
# from typing import List, Any, Dict
#
#
# import torch
# import torch.nn.functional as F
# import tqdm
# from torch.utils.data import DataLoader
#
# from torchpack.callbacks.callback import Callback, Callbacks
# from torchpack.utils import humanize
# from torchpack.utils.logging import logger
# from torchpack.utils.typing import Trainer
# from torchpack import distributed as dist
# from torchquantum.super_utils import get_named_sample_arch
# from torchquantum.utils import legalize_unitary
#
#
# __all__ = ['LegalInferenceRunner', 'SubnetInferenceRunner', 'NLLError',
#            'TrainerRestore', 'MinError', 'AddNoiseInferenceRunner', 'GradRestore']
#
#
# class LegalInferenceRunner(Callback):
#     """
#     A callback that runs inference with a list of :class:`Callback`.
#     """
#     def __init__(self, dataflow: DataLoader, *,
#                  callbacks: List[Callback]) -> None:
#         self.dataflow = dataflow
#         self.callbacks = Callbacks(callbacks)
#
#     def _set_trainer(self, trainer: Trainer) -> None:
#         self.callbacks.set_trainer(trainer)
#
#     def _trigger_epoch(self) -> None:
#         self._trigger()
#
#     def _trigger(self) -> None:
#         start_time = time.perf_counter()
#         self.callbacks.before_epoch()
#
#         with torch.no_grad():
#             self.trainer.legalized_model = copy.deepcopy(self.trainer.model)
#             legalize_unitary(self.trainer.legalized_model)
#             for feed_dict in tqdm.tqdm(self.dataflow, ncols=0):
#                 self.callbacks.before_step(feed_dict)
#                 output_dict = self.trainer.run_step(feed_dict, legalize=True)
#                 self.callbacks.after_step(output_dict)
#
#         self.callbacks.after_epoch()
#         logger.info('Inference finished in {}.'.format(
#             humanize.naturaldelta(time.perf_counter() - start_time)))
#
#
# class SubnetInferenceRunner(Callback):
#     """
#     A callback that runs inference with a list of :class:`Callback`.
#     sample a subnet and run during supernet training
#     """
#     def __init__(self, dataflow: DataLoader, *,
#                  callbacks: List[Callback],
#                  subnet: str) -> None:
#         self.dataflow = dataflow
#         self.callbacks = Callbacks(callbacks)
#         self.subnet = subnet
#
#     def _set_trainer(self, trainer: Trainer) -> None:
#         self.callbacks.set_trainer(trainer)
#
#     def _trigger_epoch(self) -> None:
#         self._trigger()
#
#     def _trigger(self) -> None:
#         start_time = time.perf_counter()
#         self.callbacks.before_epoch()
#
#         with torch.no_grad():
#             sample_arch = get_named_sample_arch(self.trainer.model.arch_space,
#                                                 self.subnet)
#             self.trainer.model.set_sample_arch(sample_arch)
#             for feed_dict in tqdm.tqdm(self.dataflow, ncols=0):
#                 self.callbacks.before_step(feed_dict)
#                 output_dict = self.trainer.run_step(feed_dict)
#                 self.callbacks.after_step(output_dict)
#
#         self.callbacks.after_epoch()
#         logger.info('Inference finished in {}.'.format(
#             humanize.naturaldelta(time.perf_counter() - start_time)))
#
#
# class AddNoiseInferenceRunner(Callback):
#     """
#     A callback that runs inference with a list of :class:`Callback`.
#     sample noise and add to model during training
#     """
#     def __init__(self, dataflow: DataLoader, *,
#                  callbacks: List[Callback],
#                  noise_total_prob: float) -> None:
#         self.dataflow = dataflow
#         self.callbacks = Callbacks(callbacks)
#         self.noise_total_prob = noise_total_prob
#
#     def _set_trainer(self, trainer: Trainer) -> None:
#         self.callbacks.set_trainer(trainer)
#
#     def _trigger_epoch(self) -> None:
#         self._trigger()
#
#     def _trigger(self) -> None:
#         start_time = time.perf_counter()
#         self.callbacks.before_epoch()
#
#         with torch.no_grad():
#             orig_is_add_noise = self.trainer.model.nodes[
#                 0].noise_model_tq.is_add_noise
#             orig_noise_total_prob = self.trainer.model.nodes[
#                 0].noise_model_tq.noise_total_prob
#             orig_mode = self.trainer.model.nodes[0].noise_model_tq.mode
#
#             for node in self.trainer.model.nodes:
#                 node.noise_model_tq.noise_total_prob = self.noise_total_prob
#                 node.noise_model_tq.is_add_noise = True
#                 node.noise_model_tq.mode = 'train'
#
#             for feed_dict in tqdm.tqdm(self.dataflow, ncols=0):
#                 self.callbacks.before_step(feed_dict)
#                 output_dict = self.trainer.run_step(feed_dict)
#                 self.callbacks.after_step(output_dict)
#
#             for node in self.trainer.model.nodes:
#                 node.noise_model_tq.is_add_noise = orig_is_add_noise
#                 node.noise_model_tq.noise_total_prob = orig_noise_total_prob
#                 node.noise_model_tq.mode = orig_mode
#
#         self.callbacks.after_epoch()
#         logger.info('Inference finished in {}.'.format(
#             humanize.naturaldelta(time.perf_counter() - start_time)))
#
#
# class NLLError(Callback):
#     def __init__(self,
#                  *,
#                  output_tensor: str = 'outputs',
#                  target_tensor: str = 'targets',
#                  name: str = 'error') -> None:
#         self.output_tensor = output_tensor
#         self.target_tensor = target_tensor
#         self.name = name
#
#     def _before_epoch(self):
#         self.size = 0
#         self.errors = 0
#
#     def _after_step(self, output_dict: Dict[str, Any]) -> None:
#         outputs = output_dict[self.output_tensor]
#         targets = output_dict[self.target_tensor]
#
#         error = F.nll_loss(outputs, targets)
#
#         self.size += targets.size(0)
#         self.errors += error.item() * targets.size(0)
#
#     def _after_epoch(self) -> None:
#         self.size = dist.allreduce(self.size, reduction='sum')
#         self.errors = dist.allreduce(self.errors, reduction='sum')
#         self.trainer.summary.add_scalar(self.name, self.errors / self.size)
#
#
# class MinError(Callback):
#     def __init__(self,
#                  *,
#                  output_tensor: str = 'outputs',
#                  target_tensor: str = 'targets',
#                  name: str = 'error') -> None:
#         self.output_tensor = output_tensor
#         self.target_tensor = target_tensor
#         self.name = name
#
#     def _before_epoch(self):
#         self.size = 0
#         self.errors = 0
#
#     def _after_step(self, output_dict: Dict[str, Any]) -> None:
#         outputs = output_dict[self.output_tensor]
#         targets = output_dict[self.target_tensor]
#
#         error = outputs.sum()
#
#         self.size += outputs.size(0)
#         self.errors += error.item()
#
#     def _after_epoch(self) -> None:
#         self.size = dist.allreduce(self.size, reduction='sum')
#         self.errors = dist.allreduce(self.errors, reduction='sum')
#         self.trainer.summary.add_scalar(self.name, self.errors / self.size)
#
#
# class TrainerRestore(Callback):
#     def __init__(self, state) -> None:
#         self.state = state
#
#     def _before_train(self) -> None:
#         self.trainer.load_state_dict(self.state)
#
#
# class GradRestore(Callback):
#     """
#     A callback that restore the all the gradients among all the steps.
#     """
#     def __init__(self) -> None:
#         self.trainer = None
#         pass
#
#     def _set_trainer(self, trainer: Trainer) -> None:
#         self.trainer = trainer
#
#     def _trigger_step(self) -> None:
#         self._trigger()
#
#     def _trigger(self) -> None:
#         for node in self.trainer.model.nodes:
#             for i, param in enumerate(node.q_layer.parameters()):
#                 self.trainer.summary.add_scalar('grad/grad_'+str(i), float(param.grad))
#                 self.trainer.summary.add_scalar('param/param_'+str(i), float(param))
#                 # self.trainer.summary.writers[1].add_histogram('histogram/grad', float(param.grad))

import torch
import torch.nn as nn
import numpy as np

from torchquantum.macro import C_DTYPE
from torchquantum.functional import func_name_dict

from typing import Union

__all__ = ["QuantumDevice"]


class QuantumDevice(nn.Module):
    def __init__(
        self,
        n_wires: int,
        device_name: str = "default",
        bsz: int = 1,
        device: Union[torch.device, str] = "cpu",
        record_op: bool = False,
    ):
        """A quantum device that contains the quantum state vector.
        Args:
            n_wires: number of qubits
            device_name: name of the quantum device
            bsz: batch size of the quantum state
            device: which classical computing device to use, 'cpu' or 'cuda'
            record_op: whether to record the operations on the quantum device and then
                they can be used to construct a static computation graph
        """
        super().__init__()
        # number of qubits
        # the states are represented in a multi-dimension tensor
        # from left to right: qubit 0 to n
        self.n_wires = n_wires
        self.device_name = device_name
        self.bsz = bsz
        self.device = device

        _state = torch.zeros(2**self.n_wires, dtype=C_DTYPE)
        _state[0] = 1 + 0j  # type: ignore
        _state = torch.reshape(_state, [2] * self.n_wires).to(self.device)
        self.register_buffer("state", _state)

        repeat_times = [bsz] + [1] * len(self.state.shape)  # type: ignore
        self._states = self.state.repeat(*repeat_times)  # type: ignore
        self.register_buffer("states", self._states)

        self.record_op = record_op
        self.op_history = []

    def reset_op_history(self):
        self.op_history = []

    def clone_states(self, existing_states: torch.Tensor):
        """Clone the states of the quantum device."""
        self.states = existing_states.clone()

    def set_states(self, states: torch.Tensor):
        """Set the states of the quantum device. The states are represented"""
        bsz = states.shape[0]
        self.states = torch.reshape(states, [bsz] + [2] * self.n_wires)

    def reset_states(self, bsz: int):
        repeat_times = [bsz] + [1] * len(self.state.shape)
        self.states = self.state.repeat(*repeat_times).to(self.state.device)

    def reset_identity_states(self):
        """Make the states as the identity matrix, one dim is the batch
        dim. Useful for verification.
        """
        self.states = torch.eye(
            2**self.n_wires, device=self.state.device, dtype=C_DTYPE
        ).reshape([2**self.n_wires] + [2] * self.n_wires)

    def reset_all_eq_states(self, bsz: int):
        """Make the states as the equal superposition state, one dim is the
        batch dim. Useful for verification.
        """
        energy = np.sqrt(1 / (2**self.n_wires) / 2)
        all_eq_state = torch.ones(2**self.n_wires, dtype=C_DTYPE) * (
            energy + energy * 1j
        )
        all_eq_state = all_eq_state.reshape([2] * self.n_wires)
        repeat_times = [bsz] + [1] * len(self.state.shape)
        self.states = all_eq_state.repeat(*repeat_times).to(self.state.device)

    def get_states_1d(self):
        """Return the states in a 1d tensor."""
        bsz = self.states.shape[0]
        return torch.reshape(self.states, [bsz, 2**self.n_wires])

    def get_state_1d(self):
        """Return the state in a 1d tensor."""
        return torch.reshape(self.state, [2**self.n_wires])

    @property
    def name(self):
        """Return the name of the device."""
        return self.__class__.__name__

    def __repr__(self):
        return f" class: {self.name} \n device name: {self.device_name} \n number of qubits: {self.n_wires} \n batch size: {self.bsz} \n current computing device: {self.state.device} \n recording op history: {self.record_op} \n current states: {self.get_states_1d().cpu().detach().numpy()}"


for func_name, func in func_name_dict.items():
    setattr(QuantumDevice, func_name, func)
