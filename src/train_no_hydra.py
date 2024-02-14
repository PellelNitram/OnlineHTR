# TODO: Implement it using PyTorch Lightning but no Hydra at first.

# Essentially, I replicate `src/train.py::train()` but without Hydra
# and thereby hard-coded settings like data and model.

import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
import torch

from src.utils import (
    RankedLogger,
)

# ==============================================
# ================== Settings ==================
# ==============================================

SEED = 42

# ==============================================
# ================== Main Code =================
# ==============================================

log = RankedLogger(__name__, rank_zero_only=True)

L.seed_everything(SEED, workers=True)

log.info(f"Instantiating datamodule")
datamodule: LightningDataModule = # TODO

log.info(f"Instantiating model")
model: LightningModule = # TODO

log.info("Instantiating callbacks...")
callbacks: List[Callback] = # TODO

log.info("Instantiating loggers...")
logger: List[Logger] = # TODO

log.info(f"Instantiating trainer")
trainer: Trainer = # TODO

# TODO