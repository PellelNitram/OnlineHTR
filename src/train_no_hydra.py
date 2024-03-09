# TODO: Implement it using PyTorch Lightning but no Hydra at first.

# Essentially, I replicate `src/train.py::train()` but without Hydra
# and thereby hard-coded settings like data and model.

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
import torch
from torch.optim import Adam
from lightning.pytorch.trainer import Trainer

from src.utils import (
    RankedLogger,
)
from src.data.online_handwriting_datasets import XournalPagewiseDatasetPyTorch
from src.data.online_handwriting_datasets import IAM_OnDB_Dataset
from src.data.online_handwriting_datasets import get_alphabet_from_dataset
from src.data.online_handwriting_datasets import get_number_of_channels_from_dataset
from src.data.online_handwriting_datamodule import SimpleOnlineHandwritingDataModule
from src.data.online_handwriting_datamodule import IAMOnDBDataModule
from src.models.carbune_module import CarbuneLitModule
from src.models.carbune_module import CarbuneLitModule2
from src.models.components.carbune2020_net import Carbune2020NetAttempt1
from src.data.transforms import TwoChannels
from src.data.tokenisers import AlphabetMapper

# ==============================================
# ================== Settings ==================
# ==============================================

SEED = 42

MAX_EPOCHS = 20_000

# ==============================================
# ================== Main Code =================
# ==============================================

log = RankedLogger(__name__, rank_zero_only=True)

L.seed_everything(SEED, workers=True)

path = '/storage/datastore-personal/s1691089/data/code/carbune2020_implementation/data/datasets/2024-01-20-xournal_dataset.xoj'
path = '/storage/datastore-personal/s1691089/data/code/carbune2020_implementation/data/datasets/2024-02-16-xournal_dataset.xoj'
output_dir = Path('/storage/datastore-personal/s1691089/data/code/carbune2020_implementation/data/train_no_hydra')

log.info(f"Instantiating datamodule")
datamodule: LightningDataModule = IAMOnDBDataModule(
    Path('data/datasets/IAM-OnDB'),
    (1_000, 0, 0),
    64,
    0,
    1000,
    False,
)

# # Code for checking `datamodule`
# datamodule.setup()
# train_dataloader = datamodule.train_dataloader()
# item = next(iter(train_dataloader))
# print(item)
# exit()

log.info(f"Instantiating model")
model: LightningModule = CarbuneLitModule2()

# log.info("Instantiating callbacks...")
# callbacks: List[Callback] = # TODO

log.info("Instantiating loggers...")
logger: List[Logger] = [
    L.pytorch.loggers.csv_logs.CSVLogger(
        save_dir=output_dir,
        name='csv/',
        prefix='',
    ),
    L.pytorch.loggers.tensorboard.TensorBoardLogger(
        save_dir=output_dir / 'tensorboard',
        name=None,
        log_graph=False,
        default_hp_metric=True,
        prefix='',
    ),
]

log.info(f"Instantiating trainer")
trainer: Trainer = Trainer(
    default_root_dir=output_dir,
    min_epochs=1, # prevents early stopping
    max_epochs=MAX_EPOCHS,
    accelerator='gpu',
    devices=1,
    check_val_every_n_epoch=1,
    deterministic=False,
    log_every_n_steps=1,
    # Added back in later:
    # callbacks=callbacks,
    logger=logger,
)

log.info("Starting training!")
trainer.fit(
    model=model,
    datamodule=datamodule,
    # ckpt_path=cfg.get("ckpt_path"),
)