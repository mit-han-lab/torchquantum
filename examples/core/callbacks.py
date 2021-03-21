import copy
import time
from typing import List

import torch
import tqdm
from torch.utils.data import DataLoader

from torchpack.callbacks.callback import Callback, Callbacks
from torchpack.utils import humanize
from torchpack.utils.logging import logger
from torchpack.utils.typing import Trainer

from torchquantum.utils import legalize_unitary


__all__ = ['LegalInferenceRunner', 'SubnetInferenceRunner']


class LegalInferenceRunner(Callback):
    """
    A callback that runs inference with a list of :class:`Callback`.
    """
    def __init__(self, dataflow: DataLoader, *,
                 callbacks: List[Callback]) -> None:
        self.dataflow = dataflow
        self.callbacks = Callbacks(callbacks)

    def _set_trainer(self, trainer: Trainer) -> None:
        self.callbacks.set_trainer(trainer)

    def _trigger_epoch(self) -> None:
        self._trigger()

    def _trigger(self) -> None:
        start_time = time.perf_counter()
        self.callbacks.before_epoch()

        with torch.no_grad():
            self.trainer.legalized_model = copy.deepcopy(self.trainer.model)
            legalize_unitary(self.trainer.legalized_model)
            for feed_dict in tqdm.tqdm(self.dataflow, ncols=0):
                self.callbacks.before_step(feed_dict)
                output_dict = self.trainer.run_step(feed_dict, legalize=True)
                self.callbacks.after_step(output_dict)

        self.callbacks.after_epoch()
        logger.info('Inference finished in {}.'.format(
            humanize.naturaldelta(time.perf_counter() - start_time)))


class SubnetInferenceRunner(Callback):
    """
    A callback that runs inference with a list of :class:`Callback`.
    sample a subnet and run during supernet training
    """
    def __init__(self, dataflow: DataLoader, *,
                 callbacks: List[Callback],
                 subnet: str) -> None:
        self.dataflow = dataflow
        self.callbacks = Callbacks(callbacks)
        self.subnet = subnet

    def _set_trainer(self, trainer: Trainer) -> None:
        self.callbacks.set_trainer(trainer)

    def _trigger_epoch(self) -> None:
        self._trigger()

    def _trigger(self) -> None:
        start_time = time.perf_counter()
        self.callbacks.before_epoch()

        with torch.no_grad():
            sample_config = \
                self.trainer.config_sampler.get_named_sample_config(
                    self.subnet)
            self.trainer.model.set_sample_config(sample_config)
            for feed_dict in tqdm.tqdm(self.dataflow, ncols=0):
                self.callbacks.before_step(feed_dict)
                output_dict = self.trainer.run_step(feed_dict)
                self.callbacks.after_step(output_dict)

        self.callbacks.after_epoch()
        logger.info('Inference finished in {}.'.format(
            humanize.naturaldelta(time.perf_counter() - start_time)))
