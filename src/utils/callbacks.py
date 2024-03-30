import time

from lightning.pytorch.callbacks import Callback


class TrainEpochTimeMeasuring(L.Callback):
    """Measures train epoch compute time."""

    def __init__(self):
        self.time_cache = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.time_cache = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        time_difference = time.perf_counter() - self.time_cache
        pl_module.log('epoch_time', time_difference, pl_module.current_epoch)
        # TODO: Potential problem are validation steps as they are part of the training loop.
        #       Should I maybe instead track batch times instead as they do not include the validation loop.
        #       See here: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks.
