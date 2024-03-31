import time

from lightning.pytorch.callbacks import Callback


class MeasureSpeed(Callback):
    """Measures train epoch compute time."""

    def __init__(self):
        self.time_cache = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.time_cache = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        time_difference = time.perf_counter() - self.time_cache
        pl_module.log(
            'batch_time',
            time_difference,
            on_step=True,
            on_epoch=False,
            prog_bar=False
        )